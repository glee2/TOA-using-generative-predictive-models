# Notes
'''
Author: Gyumin Lee
Version: 2.0
Description (primary changes): Code refactoring
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/class2class/Tech_Gen/'
import sys
sys.path.append(root_dir)

import os
import copy
import time
import re
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

''' To use PyTorch-Encoding parallel codes '''
from parallel import DataParallelModel, DataParallelCriterion

from accelerate import Accelerator
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, classification_report

from models import VCLS2CLS
from utils import print_gpu_memcheck, to_device, loss_KLD, KLDLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ON_IPYTHON = True
try:
    get_ipython()
except:
    ON_IPYTHON = False

model_dir = os.path.join(root_dir, "models")    
    
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

    def __call__(self, model, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, val_loss, self.path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                # Load latest saved model
                model = self.load_model(model)
        else:
            self.best_score = score
            self.save_checkpoint(model, val_loss, self.path)
            self.counter = 0

        return model

    def save_checkpoint(self, model, val_loss, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

    def load_model(self, model):
        saved_model = copy.copy(model)
        saved_model.load_state_dict(torch.load(self.path))
        return saved_model

# class EarlyStopping_multi(EarlyStopping):
class EarlyStopping_multi:
    def __init__(self, patience=10, verbose=True, delta=0, path='../models/checkpoint.ckpt', criteria=None, alternate_train_threshold=None, model_params={}):
        print("ES, patience:",patience)
#         super().__init__()
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.model_params = model_params
        self.best_scores = None
        self.alternate_train_threshold = alternate_train_threshold
        if delta is not None:
            self.delta = delta
        else:
            self.delta = 0.01
        self.deltas = None
        self.loss_types = ["total", "recon", "kld", "y"]
        if criteria is not None:
            self.criteria = criteria
        else:
            self.criteria = ["recon", "y"]
        self.val_losses_min = {criterion: np.Inf for criterion in self.criteria}

    def __call__(self, model, val_losses={}, ep=None):
        scores = {k: -val_losses[k] for k in self.loss_types}
        
        if (self.alternate_train_threshold is not None and ep > self.alternate_train_threshold) | (self.alternate_train_threshold is None):
            print("alternate_train_threshold entered")
            self.change_criteria(["recon", "y"])
        else:
            self.change_criteria(["recon"])

        if self.best_scores is None:
            self.best_scores = scores
            self.deltas = {k: self.best_scores[k]*self.delta for k in self.loss_types}
            self.save_checkpoint(model, val_losses, ep, self.path, show_loss=True)
        elif any([scores[k] < self.best_scores[k] + self.deltas[k] for k in self.criteria]):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
                # Load latest saved model
                model = self.load_model(model)
        else:
            self.best_scores = scores
            self.deltas = {k: self.best_scores[k]*self.delta for k in self.loss_types}
            self.save_checkpoint(model, val_losses, ep, self.path, show_loss=True)
            self.counter = 0
            
        return model

    def load_model(self, model):
        saved_model = copy.copy(model)
        saved_model.load_state_dict(torch.load(self.path))
        return saved_model
    
    def change_criteria(self, criteria):
        self.criteria = criteria
        
    def show_verbose(self, val_losses):
        if self.verbose:
            for criterion in self.criteria:
                print(f'Validation loss[{criterion}] decreased ({self.val_losses_min[criterion]:.6f} --> {val_losses[criterion]:.6f})')
            print("\n")

    def save_checkpoint(self, model, val_losses, ep, path, show_loss=False):
        if self.verbose and show_loss:
            self.show_verbose(val_losses)
        torch.save(model.state_dict(), path)
        self.val_losses_min = val_losses

def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def kl_anneal_function(step, func_type="logistic", k=0.0025, x0=2500):
    if func_type == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif func_type == 'linear':
        return min(1, step/x0)

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

    model = VCLS2CLS(model_params, tokenizers=tokenizers).to(device)
    if re.search("^2.", torch.__version__) is not None:
        print("INFO: PyTorch 2.* imported, compile model")
        model = torch.compile(model)

    if len(device_ids) > 1 and not model_params['use_accelerator']:
        model = DataParallelModel(model, device_ids=device_ids)

    return model

def train_model(model, train_loader, val_loader, model_params={}, train_params={}, class_weights=None, trial=None, model_config_name_prefix=""):
    device = model_params['device']
    writer = SummaryWriter(log_dir=os.path.join(train_params['root_dir'], "results", "TB_logs"))
    if train_params['use_accelerator']:
        accelerator = train_params['accelerator']

    if train_params["alternate_train"]:
        if train_params["alternate_train_threshold"] is not None:
            alternate_train_threshold = train_params["alternate_train_threshold"]
        else:
            alternate_train_threshold = min(math.floor(train_params["max_epochs"] * 0.4), 10)
        print("alternate_train_threshold: {}".format(alternate_train_threshold))
    else: alternate_train_threshold = None

    ## Loss function and optimizers
    loss_recon = torch.nn.CrossEntropyLoss(ignore_index=model_params['i_padding'], reduction="sum")
    loss_y = torch.nn.MSELoss() if model_params['n_outputs']==1 else torch.nn.NLLLoss(weight=class_weights, reduction="sum")
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
        if model_params["model_type"]!="enc-pred-dec":
            early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=os.path.join(train_params['root_dir'], "models/ES_checkpoint.ckpt"), model_params=model_params)
        else:
            early_stopping = EarlyStopping_multi(patience=early_stop_patience, verbose=True, path=os.path.join(train_params['root_dir'], "models/ES_checkpoint.ckpt"), alternate_train_threshold=alternate_train_threshold, model_params=model_params)

    print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Before training")

    if model_params["pretrained_enc"]: model.module.freeze(module_name="claim_encoder")

    for ep in range(train_params["curr_ep"], max_epochs+1):
        epoch_start = time.time()
        print("Epoch {} / {}\n".format(ep, train_params["max_epochs"])+str("-"*25))

        if model_params["model_type"]=="enc-pred-dec" and train_params["alternate_train"]:
            if ep > alternate_train_threshold:
                # model.module.freeze(module_name="decoder")
                model.module.freeze(module_name="predictor", defreeze=True)
            else:
                model.module.freeze(module_name="decoder", defreeze=True)
                model.module.freeze(module_name="predictor")

        train_loss = run_epoch(train_loader, model, epoch=ep, loss_f=loss_f, optimizer=optimizer, mode='train', train_params=train_params, model_params=model_params, alternate_train_threshold=alternate_train_threshold)
        val_loss = run_epoch(val_loader, model, epoch=ep, loss_f=loss_f, optimizer=optimizer, mode='eval', train_params=train_params, model_params=model_params, alternate_train_threshold=alternate_train_threshold)
        epoch_end = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start, epoch_end)
        print(f'Epoch: {ep + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f"Avg train loss: {train_loss['total']:>5f} (recon loss: {train_loss['recon']:>5f}, kld loss: {train_loss['kld']:>5f}, y loss: {train_loss['y']:>5f})\nAvg val loss: {val_loss['total']:>5f} (recon loss: {val_loss['recon']:>5f}, kld loss: {val_loss['kld']:>5f}, y loss: {val_loss['y']:>5f})\n")

        writer.add_scalar("Loss/train[total]", train_loss["total"], ep)
        writer.add_scalar("Loss/train[recon]", train_loss["recon"], ep)
        writer.add_scalar("Loss/train[kld]", train_loss["kld"], ep)
        writer.add_scalar("Loss/train[y]", train_loss["y"], ep)
        writer.add_scalar("Loss/val[total]", val_loss["total"], ep)
        writer.add_scalar("Loss/val[recon]", val_loss["recon"], ep)
        writer.add_scalar("Loss/val[kld]", val_loss["kld"], ep)
        writer.add_scalar("Loss/val[y]", val_loss["y"], ep)
        
        train_params.update({"curr_ep": ep})

        # Intermediate save, every 2 epochs
        if ep % 2 == 0:
            model_config_name = "" + model_config_name_prefix
            key_components = {"data": ["class_level", "class_system", "max_seq_len_class", "max_seq_len_claim", "vocab_size"], "model": ["n_layers", "d_hidden", "d_pred_hidden", "d_latent", "d_embedding", "d_ff", "n_head", "d_head"], "train": ["learning_rate", "batch_size", "max_epochs", "curr_ep"]}
            model_config_name += "[{}]system".format(train_params["class_system"])
            for component in key_components["model"]:
                model_config_name += f"[{str(model_params[component])}]{component}"
            for component in key_components["train"]:
                model_config_name += f"[{str(train_params[component])}]{component}"
            final_model_path = os.path.join(model_dir, f"[MODEL]{model_config_name}.ckpt")
            
            if model_params["model_type"]!="enc-pred-dec":
                early_stopping.save_checkpoint(model, val_losses=val_loss["total"], ep=ep, path=final_model_path)
            else:
                early_stopping.save_checkpoint(model, val_losses=val_loss, ep=ep, path=final_model_path)
                
        if train_params['use_early_stopping']:
            if model_params["model_type"]!="enc-pred-dec":
                model = early_stopping(model, val_losses=val_loss["total"], ep=ep)
            else:
                model = early_stopping(model, val_losses=val_loss, ep=ep)
            if early_stopping.early_stop:
                print("Early stopped\n")
                break
            elif ep == max_epochs-1:
                model = early_stopping.load_model(model)
        torch.cuda.empty_cache()
    writer.flush()
    writer.close()
    return (model, model_params, train_params)

def run_epoch(data_loader, model, epoch=None, loss_f=None, optimizer=None, mode='train', train_params={}, model_params={}, alternate_train_threshold=None):
    tokenizer = model.module.tokenizers["class_dec"]
    device = train_params['device']
    clip_max_norm = 1
    if mode=="train":
        model.train()
        dict_epoch_losses = {"total": 0, "recon": 0, "y": 0, "kld": 0}

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])

        for step, batch_data in tqdm(enumerate(data_loader)):
            torch.cuda.empty_cache()
            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Load data")

            optimizer.zero_grad()

            if ON_IPYTHON:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, use_cuda=True) as prof:
                    with record_function("model_feedforward"):
                        outputs = model(batch_data["text_inputs"], batch_data["text_outputs"], teach_force_ratio=train_params["teach_force_ratio"]) # omit <eos> from target sequence
                        if model_params["device"] == "cpu":
                            outputs = [outputs] # for single device
                        outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                        outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                        outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                        outputs_mu = [output["mu"] for output in outputs]
                        outputs_logvar = [output["logvar"] for output in outputs]
                        dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}
            else:
                outputs = model(batch_data["text_inputs"], batch_data["text_outputs"], teach_force_ratio=train_params["teach_force_ratio"]) # omit <eos> from target sequence
                if model_params["device"] == "cpu":
                    outputs = [outputs] # for single device
                outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # change the order of class and dimension (N, d1, C) -> (N, C, d1) => outputs_recon: (batch_size, n_dec_vocab, n_dec_seq-1)
                outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                outputs_mu = [output["mu"] for output in outputs]
                outputs_logvar = [output["logvar"] for output in outputs]
                dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Forward pass")

            if "dec" in model_params["model_type"]:
                preds_recon = dict_outputs["recon"]
                trues_recon = batch_data["text_outputs"][:,1:] if model_params["model_name"] == "class2class" else batch_data["text_outputs"]["input_ids"][:,1:]
                preds_mu = dict_outputs["mu"]
                preds_logvar = torch.cat([t.to(device) for t in dict_outputs["logvar"]])
            if "pred" in model_params["model_type"]:
                preds_y = dict_outputs["y"]
                trues_y = batch_data["targets"].to(dtype=preds_y[0].dtype) if model_params["n_outputs"]==1 else batch_data["targets"]

            if model_params["model_type"] == "enc-pred-dec":
                loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                loss_kld = kl_anneal_function(step) * loss_f["KLD"](preds_mu, preds_logvar)
                loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                if train_params["alternate_train"]:
                    if epoch > alternate_train_threshold:
                        loss = loss_y + loss_recon + loss_kld
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

            dict_epoch_losses["total"] += loss.item()

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Loss calculation")

            ## averaging (by batch_size)
            loss /= data_loader.batch_size

            if train_params['use_accelerator']:
                train_params['accelerator'].backward(loss)
            else:
                loss.backward()

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Backward propagation")

            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

            optimizer.step()

            print_gpu_memcheck(verbose=train_params['mem_verbose'], devices=train_params['device_ids'], stage="Weight update")

        if ON_IPYTHON:
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
            print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

        dict_epoch_losses = {key: value / (data_loader.batch_size * len(data_loader)) for key, value in dict_epoch_losses.items()} # Averaging
    
        return dict_epoch_losses

    elif mode=="eval" or mode=="test":
        model.eval()
        dict_epoch_losses = {"total": 0, "recon": 0, "y": 0, "kld": 0}
        
        preds_y_container = []
        trues_y_container = []

        with torch.no_grad():
            for step, batch_data in tqdm(enumerate(data_loader)):
                # batch_data = to_device(batch_data, device)
                if ON_IPYTHON:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, use_cuda=True) as prof:
                        with record_function("model_feedforward"):
                            outputs = model(batch_data["text_inputs"], batch_data["text_outputs"], teach_force_ratio=0.) # omit <eos> from target sequence
                            outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                            outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                            outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                            outputs_mu = [output["mu"] for output in outputs]
                            outputs_logvar = [output["logvar"] for output in outputs]
                            # dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}
                            dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}
                else:
                    outputs = model(batch_data["text_inputs"], batch_data["text_outputs"], teach_force_ratio=0.) # omit <eos> from target sequence
                    outputs_recon = [output["dec_outputs"].permute(0,2,1)[:,:,1:] for output in outputs] # outputs_recon: n_gpus * (minibatch, n_dec_seq, n_dec_vocab)
                    outputs_z = [output["z"] for output in outputs] # outputs_z: n_gpus * (minibatch, d_hidden)
                    outputs_y = [output["pred_outputs"] for output in outputs] # outputs_y: n_gpus * (minibatch, n_outputs)
                    outputs_mu = [output["mu"] for output in outputs]
                    outputs_logvar = [output["logvar"] for output in outputs]
                    # dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z}
                    dict_outputs = {"recon": outputs_recon, "y": outputs_y, "z": outputs_z, "mu": outputs_mu, "logvar": outputs_logvar}

                if "dec" in model_params["model_type"]:
                    preds_recon = dict_outputs["recon"]
                    trues_recon = batch_data["text_outputs"][:,1:] if model_params["model_name"] == "class2class" else batch_data["text_outputs"]["input_ids"][:,1:]
                    preds_mu = dict_outputs["mu"]
                    preds_logvar = torch.cat([t.to(device) for t in dict_outputs["logvar"]])
                if "pred" in model_params["model_type"]:
                    preds_y = dict_outputs["y"]
                    preds_y_container.append(np.concatenate([ppp.cpu().detach().numpy().argmax(-1) for ppp in preds_y]))
                    trues_y = batch_data["targets"].to(dtype=preds_y[0].dtype) if model_params["n_outputs"]==1 else batch_data["targets"]
                    trues_y_container.append(np.concatenate([trues_y.cpu().detach().numpy()]))
                    

                if model_params["model_type"] == "enc-pred-dec":
                    loss_recon = train_params["loss_weights"]["recon"] * loss_f["recon"](preds_recon, trues_recon)
                    loss_kld = kl_anneal_function(step) * loss_f["KLD"](preds_mu, preds_logvar)
                    loss_y = train_params["loss_weights"]["y"] * loss_f["y"](preds_y, trues_y)
                    if train_params["alternate_train"]:
                        if epoch > alternate_train_threshold:
                            loss = loss_y + loss_recon + loss_kld
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
                    loss_y = torch.tensor(0)
                    loss = loss_recon # + loss_kld
                    dict_epoch_losses["recon"] += loss_recon.item()
                    dict_epoch_losses["kld"] += loss_kld.item()

                dict_epoch_losses["total"] += loss.item()

                ## averaging (by batch_size)
                loss /= data_loader.batch_size

        dict_epoch_losses = {key: value / (data_loader.batch_size * len(data_loader)) for key, value in dict_epoch_losses.items()} # Averaging
        
        preds_y_container = np.concatenate(preds_y_container)
        trues_y_container = np.concatenate(trues_y_container)
        preds_y_counts = (len(preds_y_container[preds_y_container==0]), len(preds_y_container[preds_y_container==1]))
        print("Predictive performance evaluation on VALIDATION_DATA\n"+classification_report(trues_y_container, preds_y_container))

        trues_class = pd.Series(tokenizer.decode_batch(trues_recon.cpu().detach().numpy()))
        preds_class = pd.Series(tokenizer.decode_batch(np.concatenate([ppp.argmax(1).cpu().detach().numpy() for ppp in preds_recon])))
        
        trues_class = trues_class.apply(lambda x: x[:x.index(tokenizer.eos_token)] if tokenizer.eos_token in x else x)
        preds_class = preds_class.apply(lambda x: x[:x.index(tokenizer.eos_token)] if tokenizer.eos_token in x else x)

        temp_recon_df = pd.concat([trues_class, preds_class], axis=1)
        temp_recon_df.columns = ['Origin Classes', 'Generated Classes']
    
        Jaccard_similarities = temp_recon_df.apply(lambda x: len(set(x["Origin Classes"]).intersection(set(x["Generated Classes"]))) / len(set(x["Origin Classes"]).union(set(x["Generated Classes"]))), axis=1)
        mean_Jaccard = np.round(np.mean(Jaccard_similarities.values),4)
        print("Generative performance evaluation on VALIDATION_DATA\n"+"avg. Jaccard similarity: {}".format(mean_Jaccard))


        return dict_epoch_losses

    else:
        print("mode is not specified")
        return

# def validate_model_mp(model, val_dataset, mp=None, batch_size=None, model_params={}, train_params={}):
def validate_model_mp(model_ckpt, val_dataset, mp=None, batch_size=None, model_params={}, train_params={}, tokenizers={}, use_gpu=False, n_cores=4):
    if batch_size is None:
        batch_size = train_params["batch_size"]
    queue_dataloader = mp.Queue()
    manager = mp.Manager()
    processes = []

    if use_gpu:
        device_list = train_params["device_ids"]
    else:
        device_list = [torch.device(type="cpu", index=i) for i in np.arange(n_cores)]
    n_devices = len(device_list)
    
#     ret_dict = manager.dict({d: manager.dict({"recon": manager.dict(), "y": manager.dict()}) for d in [device_id.index for device_id in train_params["device_ids"]]})
    ret_dict = manager.dict({d: manager.dict({"recon": manager.dict(), "y": manager.dict()}) for d in [device_id.index for device_id in device_list]})
       
    chunks = list(torch.arange(len(val_dataset)).chunk(n_devices))
    for device_id, idx_chunk in zip(device_list, chunks):
        p = mp.Process(
            name="Subprocess",
            target=inference_mp,
            args=(model_ckpt, device_id.index, val_dataset, idx_chunk, ret_dict, model_params, batch_size, tokenizers, use_gpu)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return ret_dict

def inference_mp(model_ckpt, device_rank, val_dataset, idx_chunk, ret_dict, model_params, batch_size, tokenizers, use_gpu):
    import torch
    import numpy as np
    from tqdm import tqdm
    
    root_device = torch.device("cuda") if use_gpu else torch.device("cpu")
    curr_device = torch.device(f"cuda:{device_rank}") if use_gpu else torch.device(f"cpu:{device_rank}")

    if not hasattr(tokenizers["claim_enc"], "vocab_size"):
        tokenizers["claim_enc"].vocab_size = tokenizers["claim_enc"].get_vocab_size()
    
    # 모델 로드
    model = VCLS2CLS(model_params, tokenizers=tokenizers)
    if os.path.exists(model_ckpt):
        best_states = torch.load(model_ckpt, map_location=root_device)
    else:
        raise Exception("Model need to be trained first")
        
    has_module_prefix = any(k.startswith("module.") for k in best_states.keys())
    if has_module_prefix:
        stripped = {}
        for k, v in best_states.items():
            new_key = k[len("module."):] if k.startswith("module.") else k
            stripped[new_key] = v
        best_states = stripped       
    
    model.load_state_dict(best_states)
    model = model.to(device=curr_device)
    model.device = curr_device
    model.eval()
    
    with torch.no_grad():
        try:
            subset = Subset(val_dataset, idx_chunk)
            data_loader = DataLoader(subset, batch_size=batch_size, num_workers=0, pin_memory=False)
        except TimeoutError:
            print(f"[Rank {device_rank}] queue is empty after timeout")
            return
        except Exception as e:
            print(f"[Rank {device_rank}] unexpected error: {e!r}")
            return
            
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

        trues_claims = pd.Series(configs.model.tokenizers["claim_dec"].decode_batch(recon_kw))
        trues_class = pd.Series(tokenizer.decode_batch(trues)).apply(lambda x: x[x.index(tokenizer.sos_token)+1:x.index(tokenizer.eos_token)])
        preds_class = pd.Series(tokenizer.decode_batch(preds)).apply(lambda x: x[x.index(tokenizer.sos_token)+1:x.index(tokenizer.eos_token)] if tokenizer.eos_token in x else x[x.index(tokenizer.sos_token)+1:])
        BLEU_scores = pd.Series([sentence_bleu([t],p, weights=(1.0,)) for t,p in zip(trues_class.values, preds_class.values)])
        if recon_kw is not None:
            trues_claims_kw = pd.Series(configs.model.tokenizers["claim_dec"].decode_batch(recon_kw))
            eval_res = pd.concat([trues_claims_kw, trues_class, preds_class, BLEU_scores], axis=1)
            eval_res.columns = ['Origin claims (keywords)', 'Origin Classes', 'Generated Classes', "BLEU Score"]
            Jaccard_similarities = eval_res.apply(lambda x: len(set(x["Origin Classes"]).intersection(set(x["Generated Classes"]))) / len(set(x["Origin Classes"]).union(set(x["Generated Classes"]))), axis=1)
            eval_res.loc[:,"Jaccard Similarity"] = Jaccard_similarities
            eval_res.loc[len(eval_res)] = ["", "", "Average", np.round(np.mean(BLEU_scores.values),4), np.round(np.mean(Jaccard_similarities.values),4)]
        else:
            eval_res = pd.concat([trues_claims, preds_claims, BLEU_scores], axis=1)
            eval_res.columns = ['Origin Classes', 'Origin claims', 'Generated Classes', "BLEU Score"]
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
