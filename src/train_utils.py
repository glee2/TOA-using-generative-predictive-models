# Notes
'''
Author: Gyumin Lee
Version: 0.62
Description (primary changes): Modify classification task
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/Transformer/Tech_Gen/'
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
# import torch.multiprocessing as mp
# mp.set_start_method("spawn")
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
from models import Transformer
from utils import token2class, print_gpu_memcheck

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
                saved_model = copy.copy(model)
                saved_model.load_state_dict(torch.load(self.path))
                return saved_model
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

def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def build_model(model_params={}, trial=None, tokenizer=None):
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

    model = Transformer(model_params, tokenizer=tokenizer).to(device)
    if re.search("^2.", torch.__version__) is not None:
        print("INFO: PyTorch 2.* imported, compile model")
        model = torch.compile(model)

    if len(device_ids) > 1 and not model_params['use_accelerator']:
        # model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[1])
        model = DataParallelModel(model, device_ids=device_ids)

    return model

def train_model(model, train_loader, val_loader, model_params={}, train_params={}, trial=None):
    device = model_params['device']
    writer = SummaryWriter(log_dir=os.path.join(train_params['root_dir'], "results", "TB_logs"))
    if train_params['use_accelerator']:
        accelerator = train_params['accelerator']

    ## Loss function and optimizers
    loss_recon = torch.nn.CrossEntropyLoss(ignore_index=model_params['i_padding'])
    loss_y = torch.nn.MSELoss() if model_params['n_outputs']==1 else torch.nn.CrossEntropyLoss()

    loss_recon = DataParallelCriterion(loss_recon, device_ids=model_params['device_ids'])
    loss_y = DataParallelCriterion(loss_y, device_ids=model_params['device_ids'])

    if model_params["model_type"] == "enc-pred-dec":
        loss_f = {"recon": loss_recon, "y": loss_y}
    elif model_params["model_type"] == "enc-dec":
        loss_f = {"recon": loss_recon}
        # for p in model.module.predictor.parameters():
        #     p.requires_grad = False
    elif model_params["model_type"] == "enc-pred":
        loss_f = {"y": loss_y}
        # for p in model.module.decoder.parameters():
        #     p.requires_grad = False
    else:
        loss_f = None

    if trial is not None:
        train_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        max_epochs = train_params['max_epochs_for_tune']
        early_stop_patience = train_params['early_stop_patience_for_tune']
    else:
        max_epochs = train_params['max_epochs']
        early_stop_patience = train_params['early_stop_patience']

    # optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['learning_rate'])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])

    ## Accelerator wrapping
    if train_params['use_accelerator']:
        model, train_loader, val_loader, optimizer = accelerator.prepare(model, train_loader, val_loader, optimizer)

    ## Training
    if train_params['use_early_stopping']:
        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=os.path.join(train_params['root_dir'],"models/ES_checkpoint_"+train_params['config_name']+".ckpt"))

    print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Before training")

    for ep in range(max_epochs):
        epoch_start = time.time()
        print(f"Epoch {ep+1}\n"+str("-"*25))

        if train_params["alternate_train"]:
            if ep % 2 == 0:
                for p in model.module.decoder.parameters():
                    p.requires_grad = False
            else:
                for p in model.module.decoder.parameters():
                    p.requires_grad = True
                for p in model.module.decoder.pos_emb.parameters():
                    p.requires_grad = False
                # for p in model.module.predictor.parameters():
                #     p.requires_grad = False

        train_loss = run_epoch(train_loader, model, epoch=ep, loss_f=loss_f, optimizer=optimizer, mode='train', train_params=train_params, model_params=model_params)
        val_loss = run_epoch(val_loader, model, epoch=ep, loss_f=loss_f, optimizer=optimizer, mode='eval', train_params=train_params, model_params=model_params)
        epoch_end = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start, epoch_end)
        print(f'Epoch: {ep + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f"Avg train loss: {train_loss['total']:>5f} (recon loss: {train_loss['recon']:>5f}, y loss: {train_loss['y']:>5f})\nAvg val loss: {val_loss['total']:>5f} (recon loss: {val_loss['recon']:>5f}, y loss: {val_loss['y']:>5f})\n")

        writer.add_scalar("Loss/train[total]", train_loss["total"], ep)
        writer.add_scalar("Loss/train[recon]", train_loss["recon"], ep)
        writer.add_scalar("Loss/train[y]", train_loss["y"], ep)
        writer.add_scalar("Loss/val[total]", val_loss["total"], ep)
        writer.add_scalar("Loss/val[recon]", val_loss["recon"], ep)
        writer.add_scalar("Loss/val[y]", val_loss["y"], ep)

        if train_params['use_early_stopping']:
            model = early_stopping(val_loss["total"], model)
            if early_stopping.early_stop:
                print("Early stopped\n")
                break
        torch.cuda.empty_cache()
    writer.flush()
    writer.close()
    return model

def run_epoch(data_loader, model, epoch=None, loss_f=None, optimizer=None, mode='train', train_params={}, model_params={}):
    device = train_params['device']
    clip_max_norm = 1
    if mode=="train":
        model.train()
        dict_epoch_losses = {"total": 0, "recon": 0, "y": 0}

        for i, (X, Y) in tqdm(enumerate(data_loader)):
            src, trg, y = X.to(device), X.to(device), Y.to(device)

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Load data")

            optimizer.zero_grad()

            if ON_IPYTHON:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, use_cuda=True) as prof:
                    with record_function("model_feedforward"):
                        outputs = model(src, trg[:,:-1]) # omit <eos> from target sequence
                        outputs_recon = [output[0] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                        outputs_z = [output[-2] for output in outputs]
                        outputs_y = [output[-1] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                        dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}
            else:
                outputs = model(src, trg[:,:-1]) # omit <eos> from target sequence
                outputs_recon = [output[0] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                outputs_z = [output[-2] for output in outputs]
                outputs_y = [output[-1] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Forward pass")

            if "dec" in model_params["model_type"]:
                preds_recon = [output.permute(0,2,1) for output in dict_outputs["recon"]] # change the order of class and dimension (N, d1, C) -> (N, C, d1) => pred_trg: (batch_size, n_dec_vocab, n_dec_seq-1)
                trues_recon = trg[:,1:] # omit <sos> from target sequence
            if "pred" in model_params["model_type"]:
                preds_y = dict_outputs["y"]
                trues_y = y.to(dtype=preds_y[0].dtype) if model_params["n_outputs"]==1 else y

            if model_params["model_type"] == "enc-pred-dec":
                loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                if train_params["alternate_train"]:
                    if epoch % 2 == 0:
                        loss = loss_y
                    else:
                        loss = loss_recon + loss_y
                else:
                    loss = loss_recon + loss_y
                dict_epoch_losses["recon"] += loss_recon.item()
                dict_epoch_losses["y"] += loss_y.item()
            elif model_params["model_type"] == "enc-pred":
                loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                loss = loss_y
                dict_epoch_losses["y"] += loss_y.item()
            elif model_params["model_type"] == "enc-recon":
                loss_recon = loss_f["recon"](preds_recon, trues_recon)
                loss = loss_recon
                dict_epoch_losses["recon"] += loss_recon.item()

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
        dict_epoch_losses = {"total": 0, "recon": 0, "y": 0}

        with torch.no_grad():
            for i, (X, Y) in enumerate(data_loader):
                src, trg = X.to(device), X.to(device)
                y = Y.to(device)

                outputs = model(src, trg[:,:-1]) # omit <eos> from target sequence
                outputs_recon = [output[0] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                outputs_z = [output[-2] for output in outputs]
                outputs_y = [output[-1] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}

                if "dec" in model_params["model_type"]:
                    preds_recon = [output.permute(0,2,1) for output in dict_outputs["recon"]] # change the order of class and dimension (N, d1, C) -> (N, C, d1) => pred_trg: (batch_size, n_dec_vocab, n_dec_seq-1)
                    trues_recon = trg[:,1:] # omit <sos> from target sequence
                if "pred" in model_params["model_type"]:
                    preds_y = dict_outputs["y"]
                    trues_y = y

                if model_params["model_type"] == "enc-pred-dec":
                    loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                    loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                    if train_params["alternate_train"]:
                        if epoch % 2 == 0:
                            loss = loss_y
                        else:
                            loss = loss_recon + loss_y
                    else:
                        loss = loss_recon + loss_y
                    dict_epoch_losses["recon"] += loss_recon.item()
                    dict_epoch_losses["y"] += loss_y.item()
                elif model_params["model_type"] == "enc-pred":
                    loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                    loss = loss_y
                    dict_epoch_losses["y"] += loss_y.item()
                elif model_params["model_type"] == "enc-recon":
                    loss_recon = loss_f["recon"](preds_recon, trues_recon)
                    loss = loss_recon
                    dict_epoch_losses["recon"] += loss_recon.item()

                # epoch_loss += loss.item()
                dict_epoch_losses["total"] += loss.item()

        dict_epoch_losses = {key: (value/len(data_loader)) for key, value in dict_epoch_losses.items()} # Averaging

        return dict_epoch_losses

    else:
        print("mode is not specified")
        return

def validate_model(model, val_loader, model_params={}, train_params={}):
    with torch.no_grad():
        trues_recon_val, trues_y_val = [], []
        preds_recon_val, preds_y_val = [], []
        for batch, (X_batch, Y_batch) in tqdm(enumerate(val_loader)):
            trues_recon_val.append(X_batch[:,1:].cpu().detach().numpy()) # omit <SOS>
            trues_y_val.append(Y_batch.cpu().detach().numpy())

            enc_inputs = X_batch.to(device=model_params['device'])
            enc_outputs, *_ = model.module.encoder(enc_inputs)

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Encoding done")

            preds_recon_batch = torch.tile(torch.tensor(model_params['tokenizer'].token_to_id("<SOS>"), device=model_params['device']), dims=(X_batch.shape[0],1)).to(device=model_params['device'])

            pred_outputs = model.module.predictor(enc_outputs) # pred_outputs: (batch_size, n_outputs)

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Prediction done")

            preds_y_batch = pred_outputs.cpu().detach().numpy()

            for i in range(model_params['n_dec_seq']-1):
                dec_outputs, *_ = model.module.decoder(preds_recon_batch, enc_inputs, enc_outputs)
                pred_tokens = dec_outputs.argmax(2)[:,-1].unsqueeze(1)
                preds_recon_batch = torch.cat([preds_recon_batch, pred_tokens], axis=1)
                print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Decoding for seq index "+str(i))

            preds_recon_val.append(preds_recon_batch[:,1:].cpu().detach().numpy()) # omit <SOS>
            preds_y_val.append(preds_y_batch)
    return [np.concatenate(x) for x in [trues_recon_val, preds_recon_val, trues_y_val, preds_y_val]]

def validate_model_mp(model, val_dataset, mp=None, batch_size=None, model_params={}, train_params={}):
    if batch_size is None:
        batch_size = train_params["batch_size"]
    queue_dataloader = mp.Queue()
    manager = mp.Manager()
    ret_dict = manager.dict({d: manager.dict({"recon": manager.dict(), "y": manager.dict()}) for d in range(train_params["n_gpus"])})
    processes = []

    for device_rank in range(train_params["n_gpus"]):
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
    with torch.no_grad():
        data_loader = queue_dataloader.get()
        trues_recon, trues_y = [], []
        preds_recon, preds_y = [], []

        for batch, (X_batch, Y_batch) in tqdm(enumerate(data_loader)):
            trues_recon.append(X_batch[:,1:].cpu().detach().numpy())
            trues_y.append(Y_batch.cpu().detach().numpy())

            enc_inputs = X_batch.to(device=curr_device)
            enc_outputs, *_ = model.encoder(enc_inputs)

            preds_recon_batch = torch.tile(torch.tensor(model_params['tokenizer'].token_to_id("<SOS>"), device=curr_device), dims=(X_batch.shape[0],1)).to(device=curr_device)
            preds_y_batch = model.predictor(enc_outputs) # pred_outputs: (batch_size, n_outputs)

            for i in range(model_params['n_dec_seq']-1):
                dec_outputs, *_ = model.decoder(preds_recon_batch, enc_inputs, enc_outputs)
                pred_tokens = dec_outputs.argmax(2)[:,-1].unsqueeze(1)
                preds_recon_batch = torch.cat([preds_recon_batch, pred_tokens], axis=1)

            preds_recon.append(preds_recon_batch[:,1:].cpu().detach().numpy())
            preds_y.append(preds_y_batch.cpu().detach().numpy())

        trues_recon = np.concatenate(trues_recon)
        trues_y = np.concatenate(trues_y)
        preds_recon = np.concatenate(preds_recon)
        preds_y = np.concatenate(preds_y)

        ret_dict[device_rank]["recon"] = {"true": trues_recon, "pred": preds_recon} # omit <SOS>
        ret_dict[device_rank]["y"] = {"true": trues_y, "pred": preds_y}

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

def perf_eval(model_name, trues, preds, configs=None, pred_type='regression', tokenizer=None):
    if pred_type == 'classification':
        if trues.shape != preds.shape:
            preds = preds.argmax(-1)
        metric_list = ['Support', 'Accuracy', 'Recall', 'Precision', 'F1 score', 'Specificity', 'NPV']

        cm = confusion_matrix(trues, preds)

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

        conf_mat_res = conf_mat_res = pd.DataFrame(cm, index=["True "+str(i) for i in range(configs.model.n_outputs)], columns=["Predicted "+str(i) for i in range(configs.model.n_outputs)])
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

        trues_claims = pd.Series(tokenizer.decode_batch(trues))
        preds_claims = pd.Series(tokenizer.decode_batch(preds, skip_special_tokens=False)).apply(lambda x: x.split("<EOS>")[0])
        BLEU_scores = pd.Series([sentence_bleu([t],p) for t,p in zip(trues_claims.values, preds_claims.values)])

        eval_res = pd.concat([trues_claims, preds_claims, BLEU_scores], axis=1)
        eval_res.columns = ['Origin SEQ', 'Generated SEQ', "BLEU Score"]
        eval_res.loc[len(eval_res)] = ["", "Average BLEU Score", np.round(np.mean(BLEU_scores.values),4)]

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

'''
[REFERENCE]
def run_epoch(dataloader, model, loss_recon, loss_y, mode='train', optimizer=None, loss_weights={'recon': 5, 'y': 5}, device='cpu'):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    test_loss, correct = 0, 0
    loss_weights['recon'] = loss_weights['recon'] / sum(loss_weights.values())
    loss_weights['y'] = 1 - loss_weights['recon']

    loss_out = {'total': [], 'recon': [], 'y': []}
    # batch_losses = []
    if mode == 'train':
        model.train()

        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device, dtype=torch.long), Y.to(device, dtype=torch.float) # X: (batch_size, seq_len)
            preds_recon, preds_y, z = model(X) # preds_recon: (batch_size, vocab_size, seq_len), preds_y: (batch_size, 1), z: (n_layers, batch_size, hidden_dim * n_directions)
            trues_recon = X.clone()
            trues_y = Y.clone()

            batch_loss_recon = loss_weights['recon']*loss_recon(preds_recon, trues_recon)
            batch_loss_y = loss_weights['y']*loss_y(preds_y, trues_y)
            batch_loss_total = sum([batch_loss_recon, batch_loss_y])

            loss_out['recon'].append(batch_loss_recon.item())
            loss_out['y'].append(batch_loss_y.item())
            loss_out['total'].append(batch_loss_total.item())

            optimizer.zero_grad()
            # batch_loss_total.sum().backward()
            batch_loss_total.backward()
            optimizer.step()

            if batch % 10 == 0 or batch == len(dataloader)-1:
                loss, current = batch_loss_total.item(), batch*len(X)
                if batch == len(dataloader)-1:
                    current = size
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]", end='\r', flush=True)
    else:
        model.eval()
        with torch.no_grad():
            for X, Y in dataloader:
                X, Y = X.to(device, dtype=torch.long), Y.to(device, dtype=torch.long) # X: (batch_size, seq_len)
                preds_recon, preds_y, z = model(X) # preds_recon: (batch_size, vocab_size, seq_len), preds_y: (batch_size, 1), z: (n_layers, batch_size, hidden_dim * n_directions)
                trues_recon = X.clone()
                trues_y = Y.clone()

                batch_loss_recon = loss_weights['recon']*loss_recon(preds_recon, trues_recon)
                batch_loss_y = loss_weights['y']*loss_y(preds_y, trues_y)
                batch_loss_total = sum([batch_loss_recon, batch_loss_y])

                loss_out['recon'].append(batch_loss_recon.item())
                loss_out['y'].append(batch_loss_y.item())
                loss_out['total'].append(batch_loss_total.item())

    # return np.average(batch_losses)
    loss_out = {l: np.mean(loss_out[l]) for l in loss_out.keys()}
    return loss_out
'''

'''
[REFERENCE]
def validate_model(model, val_loader, model_params={}):
    trues_recon_val, trues_y_val = [], []
    preds_recon_val, preds_y_val = [], []
    for batch, (X_batch, Y_batch) in tqdm(enumerate(val_loader)):
        trues_recon_val.append(X_batch[:,1:].cpu().detach().numpy())
        # trues_y_val.append(Y_batch.cpu().detach().numpy())
        # preds_recon_batch, preds_y_batch, z_batch = model.module(X_batch.to(device=model_params['device']))
        preds_recon_batch, *_ = model.module(X_batch.to(device=model_params['device']), X_batch[:,:-1].to(device=model_params['device']))
        preds_recon_batch = preds_recon_batch.argmax(2)
        preds_recon_val.append(preds_recon_batch.cpu().detach().numpy())
        # preds_y_val.append(preds_y_batch.cpu().detach().numpy())
    # return [np.concatenate(x) for x in [trues_recon_val, trues_y_val, preds_recon_val, preds_y_val]]
    return [np.concatenate(x) for x in [trues_recon_val, preds_recon_val]]
'''

'''
[REFERENCE]
def objective(trial, train_loader, val_loader, model_params_obj={}, train_params_obj={}):
    model = build_model(model_params_obj, trial=trial)
    model = train_model(model, train_loader, val_loader, model_params_obj, train_params_obj, trial=trial)
    trues_recon_val, trues_y_val, preds_recon_val, preds_y_val = validate_model(model, val_loader, model_params=model_params_obj)

    score_mse = mean_squared_error(trues_y_val, preds_y_val)

    return score_mse
'''
