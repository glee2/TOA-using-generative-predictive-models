# Notes
'''
Author: Gyumin Lee
Version: 1.3
Description (primary changes): Claim + class -> class
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/class2class/Tech_Gen/'
import sys
sys.path.append(root_dir)

import os
import copy
import functools
import operator
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

''' To use PyTorch-Encoding parallel codes '''
from parallel import DataParallelModel, DataParallelCriterion

from accelerate import Accelerator
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, log_loss

from data import TechDataset, CVSampler
from models import Transformer, CLS2CLS
from utils import token2class, print_gpu_memcheck, to_device, loss_KLD, KLDLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ON_IPYTHON = True
try:
    get_ipython()
except:
    ON_IPYTHON = False

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path='../models/checkpoint.ckpt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                # Load latest saved model
                self.load_model(model)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return model

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_model(self, model):
        saved_model = copy.copy(model)
        saved_model.load_state_dict(torch.load(self.path))
        return saved_model

def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def build_model(model_params={}, trial=None, tokenizers=None):
    device = model_params['device']
    device_ids = model_params['device_ids']

    if trial is not None:
        model_params['n_layers'] = trial.suggest_int("n_layers", model_params["for_tune"]["min_n_layers"], model_params["for_tune"]["max_n_layers"])
        model_params['d_embedding'] = trial.suggest_categorical("d_embedding", [x for x in np.arange(model_params["for_tune"]["min_d_embedding"],model_params["for_tune"]["max_d_embedding"]+1,step=model_params["for_tune"]["min_d_embedding"]) if model_params["for_tune"]["max_d_embedding"]%x==0])
        model_params['d_hidden'] = trial.suggest_categorical("d_hidden", [x for x in np.arange(model_params["for_tune"]["min_d_hidden"],model_params["for_tune"]["max_d_hidden"]+1,step=model_params["for_tune"]["min_d_hidden"]) if model_params["for_tune"]["max_d_hidden"]%x==0])
        model_params['d_ff'] = trial.suggest_categorical("d_ff", [x for x in np.arange(model_params["for_tune"]["min_d_ff"],model_params["for_tune"]["max_d_ff"]+1,step=model_params["for_tune"]["min_d_ff"]) if model_params["for_tune"]["max_d_ff"]%x==0])
        model_params['n_head'] = trial.suggest_categorical("n_head", [x for x in np.arange(model_params["for_tune"]["min_n_head"],model_params["for_tune"]["max_n_head"]+1,step=model_params["for_tune"]["min_n_head"]) if model_params["for_tune"]["max_n_head"]%x==0])
        model_params['d_head'] = trial.suggest_categorical("d_head", [x for x in np.arange(model_params["for_tune"]["min_d_head"],model_params["for_tune"]["max_d_head"]+1,step=model_params["for_tune"]["min_d_head"]) if model_params["for_tune"]["max_d_head"]%x==0])
        model_params['d_latent'] = trial.suggest_int("d_latent", model_params['d_hidden'] * model_params['n_enc_seq'], model_params['d_hidden'] * model_params['n_enc_seq'])

    # model = Transformer(model_params, tokenizers=tokenizers).to(device)
    model = CLS2CLS(model_params, tokenizers=tokenizers).to(device)
    if re.search("^2.", torch.__version__) is not None:
        print("INFO: PyTorch 2.* imported, compile model")
        model = torch.compile(model)

    if len(device_ids) > 1 and not model_params['use_accelerator']:
        # model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[1])
        model = DataParallelModel(model, device_ids=device_ids)

    return model

def train_model(model, train_loader, val_loader, model_params={}, train_params={}, class_weights=None, trial=None):
    device = model_params['device']
    writer = SummaryWriter(log_dir=os.path.join(train_params['root_dir'], "results", "TB_logs"))
    if train_params['use_accelerator']:
        accelerator = train_params['accelerator']

    ## Loss function and optimizers
    loss_recon = torch.nn.CrossEntropyLoss(ignore_index=model_params['i_padding'], reduction="sum")
    loss_y = torch.nn.MSELoss() if model_params['n_outputs']==1 else torch.nn.NLLLoss(weight=class_weights)
    loss_kld = KLDLoss()
    loss_recon = DataParallelCriterion(loss_recon, device_ids=model_params['device_ids'])
    loss_y = DataParallelCriterion(loss_y, device_ids=model_params['device_ids'])
    loss_kld = DataParallelCriterion(loss_kld, device_ids=model_params["device_ids"])

    if model_params["model_type"] == "enc-pred-dec":
        loss_f = {"recon": loss_recon, "y": loss_y, "KLD": loss_kld}
    elif model_params["model_type"] == "enc-dec":
        loss_f = {"recon": loss_recon, "KLD": loss_KLD}
    elif model_params["model_type"] == "enc-pred":
        loss_f = {"y": loss_y}
    else:
        loss_f = None

    if trial is not None:
        train_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        max_epochs = train_params['max_epochs_for_tune']
        early_stop_patience = train_params['early_stop_patience_for_tune']
    else:
        max_epochs = train_params['max_epochs']
        early_stop_patience = train_params['early_stop_patience']

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])

    ## Accelerator wrapping
    if train_params['use_accelerator']:
        model, train_loader, val_loader, optimizer = accelerator.prepare(model, train_loader, val_loader, optimizer)

    ## Training
    if train_params['use_early_stopping']:
        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=os.path.join(train_params['root_dir'], "models/ES_checkpoint_"+train_params['config_name']+".ckpt"))

    print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Before training")

    if model_params["pretrained_enc"]: model.module.freeze(module_name="claim_encoder")
    # model.module.freeze(module_name="predictor")

    for ep in range(max_epochs):
        epoch_start = time.time()
        print(f"Epoch {ep+1}\n"+str("-"*25))

        if model_params["model_type"]=="enc-pred-dec" and train_params["alternate_train"]:
            if ep > train_params["max_epochs"] - max(20, int(train_params["max_epochs"] * 0.2)):
                model.module.freeze(module_name="decoder")
                model.module.freeze(module_name="predictor", defreeze=True)
                # # model.module.freeze(module="decoder")
                # model.module.defreeze(module="predictor")
            else:
                model.module.freeze(module_name="decoder", defreeze=True)
                model.module.freeze(module_name="predictor")
                # model.module.freeze(module="predictor")
                # model.module.defreeze(module="decoder")

        train_loss = run_epoch(train_loader, model, epoch=ep, loss_f=loss_f, optimizer=optimizer, mode='train', train_params=train_params, model_params=model_params)
        val_loss = run_epoch(val_loader, model, epoch=ep, loss_f=loss_f, optimizer=optimizer, mode='eval', train_params=train_params, model_params=model_params)
        epoch_end = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start, epoch_end)
        print(f'Epoch: {ep + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f"Avg train loss: {train_loss['total']:>5f} (recon loss: {train_loss['recon']:>5f}, y loss: {train_loss['y']:>5f})\nAvg val loss: {val_loss['total']:>5f} (recon loss: {val_loss['recon']:>5f}, y loss: {val_loss['y']:>5f})\n")
        print(f"Avg train loss: {train_loss['total']:>5f} (recon loss: {train_loss['recon']:>5f}, kld loss: {train_loss['kld']:>5f}, y loss: {train_loss['y']:>5f})\nAvg val loss: {val_loss['total']:>5f} (recon loss: {val_loss['recon']:>5f}, kld loss: {val_loss['kld']:>5f}, y loss: {val_loss['y']:>5f})\n")

        writer.add_scalar("Loss/train[total]", train_loss["total"], ep)
        writer.add_scalar("Loss/train[recon]", train_loss["recon"], ep)
        writer.add_scalar("Loss/train[kld]", train_loss["kld"], ep)
        writer.add_scalar("Loss/train[y]", train_loss["y"], ep)
        writer.add_scalar("Loss/val[total]", val_loss["total"], ep)
        writer.add_scalar("Loss/val[recon]", val_loss["recon"], ep)
        writer.add_scalar("Loss/val[kld]", val_loss["kld"], ep)
        writer.add_scalar("Loss/val[y]", val_loss["y"], ep)

        if train_params['use_early_stopping']:
            model = early_stopping(val_loss["total"], model)
            if early_stopping.early_stop:
                print("Early stopped\n")
                break
            elif ep == max_epochs-1:
                model = early_stopping.load_model(model)
        torch.cuda.empty_cache()
    writer.flush()
    writer.close()
    return model

def run_epoch(data_loader, model, epoch=None, loss_f=None, optimizer=None, mode='train', train_params={}, model_params={}):
    device = train_params['device']
    clip_max_norm = 1
    if mode=="train":
        model.train()
        dict_epoch_losses = {"total": 0, "recon": 0, "y": 0, "kld": 0}

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])

        for i, batch_data in tqdm(enumerate(data_loader)):
            torch.cuda.empty_cache()
            # batch_data = to_device(batch_data, device)
            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Load data")

            optimizer.zero_grad()

            if ON_IPYTHON:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, use_cuda=True) as prof:
                    with record_function("model_feedforward"):
                        outputs = model(batch_data["text_inputs"], batch_data["text_outputs"]) # omit <eos> from target sequence
                        outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                        outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                        outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                        outputs_mu = [output["mu"] for output in outputs]
                        outputs_logvar = [output["logvar"] for output in outputs]
                        # dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}
                        dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}
            else:
                outputs = model(batch_data["text_inputs"], batch_data["text_outputs"], teach_force_ratio=1.0) # omit <eos> from target sequence
                outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # change the order of class and dimension (N, d1, C) -> (N, C, d1) => outputs_recon: (batch_size, n_dec_vocab, n_dec_seq-1)
                outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                outputs_mu = [output["mu"] for output in outputs]
                outputs_logvar = [output["logvar"] for output in outputs]
                # dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}
                dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Forward pass")

            if "dec" in model_params["model_type"]:
                # preds_recon = [output.permute(0,2,1) for output in dict_outputs["recon"]] # change the order of class and dimension (N, d1, C) -> (N, C, d1) => pred_recon: (batch_size, n_dec_vocab, n_dec_seq-1)
                preds_recon = dict_outputs["recon"]
                trues_recon = batch_data["text_outputs"][:,1:] if model_params["model_name"] == "class2class" else batch_data["text_outputs"]["input_ids"][:,1:]
                preds_mu = dict_outputs["mu"]
                preds_logvar = torch.cat([t.to(device) for t in dict_outputs["logvar"]])
            if "pred" in model_params["model_type"]:
                preds_y = dict_outputs["y"]
                trues_y = batch_data["targets"].to(dtype=preds_y[0].dtype) if model_params["n_outputs"]==1 else batch_data["targets"]

            if model_params["model_type"] == "enc-pred-dec":
                loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                loss_kld = loss_f["KLD"](preds_mu, preds_logvar)
                loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                if train_params["alternate_train"]:
                    if epoch > train_params["max_epochs"] - max(20, int(train_params["max_epochs"] * 0.2)):
                        loss = loss_recon + loss_kld + loss_y
                    else:
                        loss = loss_recon + loss_kld
                else:
                    loss = loss_recon + loss_kld + loss_y
                dict_epoch_losses["recon"] += loss_recon.item()
                dict_epoch_losses["y"] += loss_y.item()
                dict_epoch_losses["kld"] += loss_kld.item()
            elif model_params["model_type"] == "enc-pred":
                loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                loss = loss_y
                dict_epoch_losses["y"] += loss_y.item()
            elif model_params["model_type"] == "enc-recon":
                loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                loss_kld = loss_f["KLD"](preds_mu, preds_logvar)
                loss = loss_recon + loss_kld
                dict_epoch_losses["recon"] += loss_recon.item()
                dict_epoch_losses["kld"] += loss_kld.item()

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Loss calculation")

            if train_params['use_accelerator']:
                train_params['accelerator'].backward(loss)
            else:
                loss.backward()

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Backward propagation")

            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

            optimizer.step()

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Weight update")

            dict_epoch_losses["total"] += loss.item()

        if ON_IPYTHON:
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
            print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

        dict_epoch_losses = {key: (value/len(data_loader)) for key, value in dict_epoch_losses.items()} # Averaging

        return dict_epoch_losses

    elif mode=="eval" or mode=="test":
        model.eval()
        dict_epoch_losses = {"total": 0, "recon": 0, "y": 0, "kld": 0}

        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(data_loader)):
                # batch_data = to_device(batch_data, device)
                if ON_IPYTHON:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, use_cuda=True) as prof:
                        with record_function("model_feedforward"):
                            outputs = model(batch_data["text_inputs"], batch_data["text_outputs"]) # omit <eos> from target sequence
                            outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                            outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                            outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                            outputs_mu = [output["mu"] for output in outputs]
                            outputs_logvar = [output["logvar"] for output in outputs]
                            # dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}
                            dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}
                else:
                    outputs = model(batch_data["text_inputs"], batch_data["text_outputs"]) # omit <eos> from target sequence
                    outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                    outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                    outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                    outputs_mu = [output["mu"] for output in outputs]
                    outputs_logvar = [output["logvar"] for output in outputs]
                    # dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}
                    dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}

                if "dec" in model_params["model_type"]:
                    # preds_recon = [output.permute(0,2,1) for output in dict_outputs["recon"]] # change the order of class and dimension (N, d1, C) -> (N, C, d1) => pred_trg: (batch_size, n_dec_vocab, n_dec_seq-1)
                    preds_recon = dict_outputs["recon"]
                    trues_recon = batch_data["text_outputs"][:,1:] if model_params["model_name"] == "class2class" else batch_data["text_outputs"]["input_ids"][:,1:]
                    preds_mu = dict_outputs["mu"]
                    preds_logvar = torch.cat([t.to(device) for t in dict_outputs["logvar"]])
                if "pred" in model_params["model_type"]:
                    preds_y = dict_outputs["y"]
                    trues_y = batch_data["targets"].to(dtype=preds_y[0].dtype) if model_params["n_outputs"]==1 else batch_data["targets"]

                if model_params["model_type"] == "enc-pred-dec":
                    loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                    loss_kld = loss_f["KLD"](preds_mu, preds_logvar)
                    loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                    if train_params["alternate_train"]:
                        if epoch > train_params["max_epochs"] - max(20, int(train_params["max_epochs"] * 0.2)):
                            loss = loss_recon + loss_kld + loss_y
                        else:
                            loss = loss_recon + loss_kld
                    else:
                        loss = loss_recon + loss_kld + loss_y
                    dict_epoch_losses["recon"] += loss_recon.item()
                    dict_epoch_losses["y"] += loss_y.item()
                    dict_epoch_losses["kld"] += loss_kld.item()
                elif model_params["model_type"] == "enc-pred":
                    loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                    loss_recon = torch.tensor(0)
                    loss = loss_y
                    dict_epoch_losses["y"] += loss_y.item()
                elif model_params["model_type"] == "enc-recon":
                    loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                    loss_kld = loss_f["KLD"](preds_mu, preds_logvar)
                    loss = loss_recon + loss_kld
                    loss_y = torch.tensor(0)
                    dict_epoch_losses["recon"] += loss_recon.item()
                    dict_epoch_losses["kld"] += loss_kld.item()

                dict_epoch_losses["total"] += loss_recon.item() + loss_kld.item() + loss_y.item()

        dict_epoch_losses = {key: (value/len(data_loader)) for key, value in dict_epoch_losses.items()} # Averaging

        return dict_epoch_losses

    else:
        print("mode is not specified")
        return

def validate_model_mp(model, val_dataset, mp=None, batch_size=None, model_params={}, train_params={}):
    if batch_size is None:
        batch_size = train_params["batch_size"]
    queue_dataloader = mp.Queue()
    manager = mp.Manager()
    ret_dict = manager.dict({d: manager.dict({"recon": manager.dict(), "y": manager.dict()}) for d in [device_id.index for device_id in train_params["device_ids"]]})
    processes = []

    for device_id in train_params["device_ids"]:
        device_rank = device_id.index
        model_rank = copy.deepcopy(model.module)
        p = mp.Process(name="Subprocess", target=inference_mp, args=(model_rank, device_rank, queue_dataloader, ret_dict, model_params, train_params))
        p.start()
        processes.append(p)

    data_loaders = [DataLoader(Subset(val_dataset, idx), batch_size=batch_size, num_workers=0) for idx in torch.arange(len(val_dataset)).chunk(train_params["n_gpus"])]

    for rank in range(train_params["n_gpus"]):
        queue_dataloader.put(data_loaders[rank])

    for p in processes:
        p.join()

    return ret_dict

def inference_mp(model, device_rank, queue_dataloader, ret_dict, model_params, train_params):
    import torch
    import numpy as np
    from tqdm import tqdm
    curr_device = torch.device(f"cuda:{device_rank}")
    model = model.to(device=curr_device)
    model.device = curr_device
    with torch.no_grad():
        data_loader = queue_dataloader.get()
        batch_size = data_loader.batch_size
        trues_recon, trues_recon_kw, trues_y = [], [], []
        preds_recon, preds_y = [], []

        for batch, batch_data in tqdm(enumerate(data_loader)):
            batch_data = to_device(batch_data, curr_device)

            trues_recon_batch = batch_data["text_outputs"] if model_params["model_name"] == "class2class" else batch_data["text_outputs"]["input_ids"][:,1:]
            trues_recon.append(trues_recon_batch.cpu().detach().numpy())

            trues_recon_kw_batch = batch_data["text_inputs"]["claim"]["input_ids"][:,1:] if model_params["model_name"] == "class2class" else batch_data["text_inputs"]["input_ids"][:,1:]
            trues_recon_kw.append(trues_recon_kw_batch.cpu().detach().numpy())
            trues_y.append(batch_data["targets"].cpu().detach().numpy())

            enc_outputs, z, mu, logvar = model.encode(batch_data["text_inputs"])

            if "pred" in model_params["model_type"]:
                preds_y_batch = model.predict(z)
                if len(preds_y_batch.size()) > 2 and preds_y_batch.size(1) == 1: preds_y_batch = preds_y_batch.squeeze(0)
                # print(z.shape)
                # preds_y_batch = model.predictor(enc_outputs) # pred_outputs: (batch_size, n_outputs)
                preds_y.append(preds_y_batch.cpu().detach().numpy())

            if "dec" in model_params["model_type"]:
                preds_recon_batch = model.decode(z, enc_outputs["class"], device=curr_device)
                preds_recon_batch = preds_recon_batch.argmax(2)
                preds_recon.append(preds_recon_batch.cpu().detach().numpy())

        if "pred" in model_params["model_type"]:
            trues_y = np.concatenate(trues_y)
            preds_y = np.concatenate(preds_y)
        else:
            trues_y = preds_y = None
        if "dec" in model_params["model_type"]:
            trues_recon = np.concatenate(trues_recon)
            trues_recon_kw = np.concatenate(trues_recon_kw)
            preds_recon = np.concatenate(preds_recon)
        else:
            trues_recon = preds_recon = None

        ret_dict[device_rank]["recon"] = {"true": trues_recon, "pred": preds_recon, "kw": trues_recon_kw}
        ret_dict[device_rank]["y"] = {"true": trues_y, "pred": preds_y}

def perf_eval(model_name, trues, preds, recon_kw=None, configs=None, pred_type='regression', tokenizer=None, custom_weight=None):
    if pred_type == 'classification':
        if trues.shape != preds.shape:
            preds_new = copy.deepcopy(preds)
            if custom_weight is not None:
                criterion_vec = np.ones_like(preds_new[:,1])
                criterion_vec[preds_new[:,1]<0] *= custom_weight
                criterion_vec[preds_new[:,1]>=0] *= 1/custom_weight
                preds_new[:,1] *= criterion_vec
                preds_new = preds_new.argmax(-1)
            else:
                preds_new = preds_new.argmax(-1)

        metric_list = ['Support', 'Accuracy', 'Recall', 'Precision', 'F1 score', 'Specificity', 'NPV']

        cm = confusion_matrix(trues, preds_new)

        metric_dict = {}
        for c in range(configs.model.n_outputs):
            res = decom_confmat(cm, c=c)
            metric_dict[c] = res
        cm_micro = np.array([np.sum([metric_dict[c]["components"][comp] for c in metric_dict.keys()]) for comp in ["tp","fn","fp","tn"]]).reshape(2,2)
        res_micro = {k: v for k, v in decom_confmat(cm_micro, c=0).items() if k not in ["components"]}
        res_macro = {metric: np.mean([metric_dict[c][metric] for c in metric_dict.keys()]) for metric in metric_list[1:]}
        res_macro["Support"] = res_micro["Support"]
        res_weighted = {metric: np.sum([metric_dict[c][metric]*metric_dict[c]["Support"] for c in metric_dict.keys()])/cm.sum() for metric in metric_list[1:]}
        res_weighted["Support"] = res_micro["Support"]
        metric_dict.update({"micro-averaged": res_micro, "macro-averaged": res_macro, "weighted-averaged": res_weighted})

        eval_res = pd.DataFrame.from_dict(metric_dict).T
        eval_res = eval_res[[c for c in eval_res.columns if c!="components"]]
        eval_res = eval_res.apply(lambda x: x.apply(lambda xx: np.round(xx,4)), axis=1)

        df_model_name = pd.DataFrame(np.tile([""], len(eval_res.columns))[np.newaxis,:], index=[model_name], columns=eval_res.columns)
        eval_res = pd.concat([df_model_name, eval_res])

        conf_mat_res = pd.DataFrame(cm, index=["True "+str(i) for i in range(2)], columns=["Predicted "+str(i) for i in range(2)])
        df_model_name = pd.DataFrame(np.tile([""], len(conf_mat_res.columns))[np.newaxis,:], index=[model_name], columns=conf_mat_res.columns)
        conf_mat_res = pd.concat([df_model_name, conf_mat_res])

        return (eval_res, conf_mat_res)

    elif pred_type == 'regression':
        metric_list = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']

        mae = mean_absolute_error(trues, preds)
        mape = np.mean(abs(trues-preds)/trues)
        mse = mean_squared_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        r2 = r2_score(trues, preds)

        eval_res = pd.DataFrame([[model_name, len(trues), mae, mape, mse, rmse, r2]], columns=['Model', 'Support']+metric_list).set_index('Model').apply(np.round, axis=0, decimals=4)

        return eval_res
    elif pred_type == 'generative':
        assert configs is not None, "Configuration is needed to evaulate Generative model"
        assert tokenizer is not None, "Tokenizer is needed to convert ids to tokens"

        ## Temporary -> TODO: make dictionary of claim and class
        # trues_class = pd.Series(tokenizer.decode_batch(trues)).apply(lambda x: ",".join(x).split(tokenizer.eos_token)[0][:-1])
        trues_class = pd.Series(tokenizer.decode_batch(trues)).apply(lambda x: x[x.index(tokenizer.sos_token)+1:x.index(tokenizer.eos_token)])
        trues_claims = pd.Series(configs.model.tokenizers["claim_dec"].decode_batch(recon_kw))
        # preds_class = pd.Series(tokenizer.decode_batch(preds)).apply(lambda x: ",".join(x).split(tokenizer.eos_token)[0][:-1])
        preds_class = pd.Series(tokenizer.decode_batch(preds)).apply(lambda x: x[x.index(tokenizer.sos_token)+1:x.index(tokenizer.eos_token)])
        BLEU_scores = pd.Series([sentence_bleu([t],p) for t,p in zip(trues_class.values, preds_class.values)])
        if recon_kw is not None:
            trues_claims_kw = pd.Series(configs.model.tokenizers["claim_dec"].decode_batch(recon_kw))
            eval_res = pd.concat([trues_class, trues_claims_kw, preds_class, BLEU_scores], axis=1)
            if configs.data.use_keywords:
                eval_res.columns = ['Origin IPCs', 'Origin claims (keywords)', 'Generated IPCs', "BLEU Score"]
            else:
                eval_res.columns = ['Origin IPCs', 'Origin claims', 'Generated IPCs', "BLEU Score"]
            eval_res.loc[len(eval_res)] = ["", "", "Average BLEU Score", np.round(np.mean(BLEU_scores.values),4)]
        else:
            eval_res = pd.concat([trues_claims, preds_claims, BLEU_scores], axis=1)
            eval_res.columns = ['Origin IPCs', 'Origin claims', 'Generated IPCs', "BLEU Score"]
            eval_res.loc[len(eval_res)] = ["", "", "Average BLEU Score", np.round(np.mean(BLEU_scores.values),4)]

        return eval_res
    else:
        print(f"Not implemented for {pred_type} type.")
        return

def decom_confmat(cm, c=None):
    n_outputs = len(cm)
    tp = cm[c,c]

    tn_idx = (np.repeat(np.arange(n_outputs)[np.arange(n_outputs)!=c], n_outputs-1).tolist(), np.tile(np.arange(n_outputs)[np.arange(n_outputs)!=c], n_outputs-1).tolist())
    tn = cm[tn_idx].sum()

    fn_idx = (np.repeat([c], n_outputs-1), np.arange(n_outputs)[np.arange(n_outputs)!=c])
    fn = cm[fn_idx].sum()

    fp_idx = (np.arange(n_outputs)[np.arange(n_outputs)!=c], np.repeat([c], n_outputs-1))
    fp = cm[fp_idx].sum()

    sup = np.sum(cm[c,:])
    acc = (tn+tp)/cm.sum()
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    spe = tn/(tn+fp)
    npv = tn/(tn+fn)
    f1 = 2*((pre*rec)/(pre+rec))

    output_dict = {"Support": sup, "Accuracy": acc, "Precision": pre, "Recall": rec, "F1 score": f1, "Specificity": spe, "NPV": npv, "components": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}

    return output_dict

def objective_cv(trial, dataset, cv_idx, model_params={}, train_params={}):
    loss_recon = torch.nn.CrossEntropyLoss(ignore_index=model_params['i_padding'])
    loss_y = torch.nn.MSELoss()

    loss_recon = DataParallelCriterion(loss_recon, device_ids=model_params['device_ids'])
    loss_y = DataParallelCriterion(loss_y, device_ids=model_params['device_ids'])

    if model_params['use_predictor']:
        loss_f = {"recon": loss_recon, "y": loss_y}
    else:
        loss_f = {"recon": loss_recon}

    scores, trained_models = [], []
    for fold in tqdm(range(train_params['n_folds'])):
        train_dataset = Subset(dataset, cv_idx[fold]['train'])
        val_dataset = Subset(dataset, cv_idx[fold]['val'])
        test_dataset = Subset(dataset, cv_idx[fold]['test'])
        min_batch_size = 16
        max_batch_size = 1024 if len(val_dataset) > 1024 else len(val_dataset)

        model_params_obj = copy.deepcopy(model_params)
        train_params_obj = copy.deepcopy(train_params)

        train_params_obj['batch_size'] = trial.suggest_int("batch_size", min_batch_size, max_batch_size, step=train_params['n_gpus'])

        train_loader = DataLoader(train_dataset, batch_size=train_params_obj['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=train_params_obj['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=min_batch_size, shuffle=True, num_workers=0, drop_last=False)

        model = build_model(model_params_obj, trial=trial)
        model = train_model(model, train_loader, val_loader, model_params_obj, train_params_obj, trial=trial)

        test_loss = run_epoch(test_loader, model, loss_f=loss_f, mode='test', train_params=train_params_obj, model_params=model_params_obj)

        scores.append(test_loss["y"])
        trained_models.append(model)
        torch.cuda.empty_cache()

    best_model = trained_models[np.argmin(scores)].module
    torch.save(best_model.state_dict(), os.path.join(train_params_obj['model_dir'],f"[HPARAM_TUNING]{trial.number}trial.ckpt"))
    torch.cuda.empty_cache()

    return np.mean(scores)
