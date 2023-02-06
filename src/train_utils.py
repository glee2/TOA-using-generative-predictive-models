# Notes
'''
Author: Gyumin Lee
Version: 0.5
Description (primary changes): Employ "Accelerator" from huggingface, to automate gpu-distributed training
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/master/Tech_Gen/'
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
# from accelerate import Accelerator
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, log_loss

from data import TechDataset, CVSampler
# from model import Encoder_SEQ, Attention, AttnDecoder_SEQ, SEQ2SEQ, Predictor
from models import Transformer
from utils import token2class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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

def run_epoch_temp(data_loader, model, loss_f=None, optimizer=None, mode='train', train_params={}):
    device = train_params['device']
    clip_max_norm = 1
    if mode=="train":
        model.train()
        epoch_loss = 0

        for i, (X, Y) in enumerate(data_loader):
            src, trg, y = X.to(device), X.to(device), Y.to(device)

            optimizer.zero_grad()

            pred_trg, *_ = model(src, trg[:,:-1]) # omit <eos> from target sequence
            # output: (batch_size, n_dec_seq-1, n_dec_vocab)
            output_dim = pred_trg.shape[-1]
            pred_trg = pred_trg.contiguous().view(-1, output_dim) # output: (batch_size * (n_dec_seq-1))
            # pred_trg = pred_trg.argmax(2).contiguous().to(device=device, dtype=torch.float32) # pred_trg = (batch_size * (n_dec_seq-1))
            true_trg = trg[:,1:].contiguous().view(-1) # omit <sos> from target sequence
            # true_trg = trg[:,1:].contiguous().to(device=device, dtype=torch.float32) # omit <sos> from target sequence

            loss = loss_f(pred_trg, true_trg)
            if train_params['use_accelerator']:
                train_params['accelerator'].backward(loss)
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)
    elif mode=="eval" or mode=="test":
        model.eval()
        epoch_loss = 0

        for i, (X, Y) in enumerate(data_loader):
            src, trg = X.to(device), X.to(device)
            y = Y.to(device)

            pred_trg, *_ = model(src, trg[:,:-1]) # omit <eos> from target sequence
            # output: (batch_size, n_dec_seq-1, n_dec_vocab)
            output_dim = pred_trg.shape[-1]
            pred_trg = pred_trg.contiguous().view(-1, output_dim) # output: (batch_size * (n_dec_seq-1))
            true_trg = trg[:,1:].contiguous().view(-1) # omit <sos> from target sequence

            loss = loss_f(pred_trg, true_trg)

            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    else:
        print("mode is not specified")
        return

def build_model(model_params={}, trial=None):
    device = model_params['device']
    device_ids = model_params['device_ids']

    if trial is not None:
        model_params['n_layers'] = trial.suggest_int("n_layers", 1, 3)
        model_params['d_embedding'] = trial.suggest_categorical("d_embedding", [32, 64, 128, 256])
        model_params['d_hidden'] = trial.suggest_categorical("d_hidden", [32, 64, 128, 256, 512])
        model_params['d_latent'] = trial.suggest_int("d_latent", model_params['d_hidden'] * model_params['n_layers'] * model_params['n_directions'], model_params['d_hidden'] * model_params['n_layers'] * model_params['n_directions'])

    model = Transformer(model_params).to(device)
    if re.search("^2.", torch.__version__) is not None:
        print("INFO: PyTorch 2.* imported, compile model")
        model = torch.compile(model)

    if len(device_ids) > 1 and not model_params['use_accelerator']:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    return model

def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model(model, train_loader, val_loader, model_params={}, train_params={}, trial=None):
    device = model_params['device']
    writer = SummaryWriter(log_dir=os.path.join(train_params['root_dir'], "results", "TB_logs"))
    if train_params['use_accelerator']:
        accelerator = train_params['accelerator']

    ## Loss function and optimizers
    loss_recon = torch.nn.CrossEntropyLoss(ignore_index=model_params['i_padding'])
    loss_y = torch.nn.MSELoss()

    if trial is not None:
        train_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        max_epochs = train_params['max_epochs_for_tune']
        early_stop_patience = train_params['early_stop_patience_for_tune']
    else:
        max_epochs = train_params['max_epochs']
        early_stop_patience = train_params['early_stop_patience']

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['learning_rate'])

    ## Accelerator wrapping
    if train_params['use_accelerator']:
        model, train_loader, val_loader, optimizer = accelerator.prepare(model, train_loader, val_loader, optimizer)

    ## Training
    if train_params['use_early_stopping']:
        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=os.path.join(train_params['root_dir'],"models/ES_checkpoint_"+train_params['config_name']+".ckpt"))

    for ep in range(max_epochs):
        epoch_start = time.time()
        print(f"Epoch {ep+1}\n"+str("-"*25))
        # train_loss = run_epoch(train_loader, model, loss_recon, loss_y, mode='train', optimizer=optimizer, loss_weights=train_params['loss_weights'], device=device)
        # val_loss = run_epoch(val_loader, model, loss_recon, loss_y, mode='test', loss_weights=train_params['loss_weights'], device=device)
        train_loss = run_epoch_temp(train_loader, model, loss_f=loss_recon, optimizer=optimizer, mode='train', train_params=train_params)
        val_loss = run_epoch_temp(val_loader, model, loss_f=loss_recon, optimizer=optimizer, mode='eval', train_params=train_params)
        epoch_end = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start, epoch_end)
        print(f'Epoch: {ep + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f"Avg train loss: {train_loss['total']:>5f}, Avg val loss: {val_loss['total']:>5f}\n")
        print(f"Avg train loss: {train_loss:>5f}, Avg val loss: {val_loss:>5f}\n")
        writer.add_scalar("Loss/train", train_loss, ep)
        writer.add_scalar("Loss/val", val_loss, ep)

        if train_params['use_early_stopping']:
            model = early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopped\n")
                break
    writer.flush()
    writer.close()
    return model

def validate_model(model, val_loader, model_params={}):
    trues_recon_val, trues_y_val = [], []
    preds_recon_val, preds_y_val = [], []
    for batch, (X_batch, Y_batch) in enumerate(val_loader):
        trues_recon_val.append(X_batch.cpu().detach().numpy())
        trues_y_val.append(Y_batch.cpu().detach().numpy())
        # preds_recon_batch, preds_y_batch, z_batch = model.module(X_batch.to(device=model_params['device']))
        preds_recon_batch, *_ = model.module(X_batch.to(device=model_params['device']), X_batch[:,:-1].to(device=model_params['device']))
        preds_recon_val.append(preds_recon_batch.cpu().detach().numpy())
        # preds_y_val.append(preds_y_batch.cpu().detach().numpy())
    # return [np.concatenate(x) for x in [trues_recon_val, trues_y_val, preds_recon_val, preds_y_val]]
    return [np.concatenate(x) for x in [trues_recon_val, preds_recon_val]]

def objective(trial, train_loader, val_loader, model_params_obj={}, train_params_obj={}):
    model = build_model(model_params_obj, trial=trial)
    model = train_model(model, train_loader, val_loader, model_params_obj, train_params_obj, trial=trial)
    trues_recon_val, trues_y_val, preds_recon_val, preds_y_val = validate_model(model, val_loader, model_params=model_params_obj)

    score_mse = mean_squared_error(trues_y_val, preds_y_val)

    return score_mse

def objective_cv(trial, dataset, cv_idx, model_params={}, train_params={}):
    scores, trained_models = [], []
    for fold in tqdm(range(train_params['n_folds'])):
        train_dataset = Subset(dataset, cv_idx[fold]['train'])
        val_dataset = Subset(dataset, cv_idx[fold]['val'])
        max_batch_size = 128 if len(val_dataset) > 128 else len(val_dataset)

        model_params_obj = copy.deepcopy(model_params)
        train_params_obj = copy.deepcopy(train_params)

        train_params_obj['batch_size'] = trial.suggest_int("batch_size", 16, max_batch_size, step=train_params['n_gpus'])

        train_loader = DataLoader(train_dataset, batch_size=train_params_obj['batch_size'], shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=train_params_obj['batch_size'], shuffle=True, num_workers=4, drop_last=True)

        # score_mse = objective(trial, train_loader, val_loader, model_params_obj=model_params_obj, train_params_obj=train_params_obj)

        model = build_model(model_params_obj, trial=trial)
        model = train_model(model, train_loader, val_loader, model_params_obj, train_params_obj, trial=trial)
        # trues_recon_val, trues_y_val, preds_recon_val, preds_y_val = validate_model(model, val_loader, model_params=model_params_obj)
        trues_recon_val, preds_recon_val = validate_model(model, val_loader, model_params=model_params_obj)

        # score_mse = mean_squared_error(trues_y_val, preds_y_val)

        # scores.append(score_mse)

        ## Cross entropy loss
        trues_recon_val = torch.tensor(trues_recon_val[:,1:]).contiguous().to(device=model_params['device'], dtype=torch.float32)
        preds_recon_val = torch.tensor(preds_recon_val).argmax(2).contiguous().to(device=model_params['device'], dtype=torch.float32)
        score_ce = F.cross_entropy(trues_recon_val, preds_recon_val).cpu().numpy()
        scores.append(score_ce)

        trained_models.append(model)

    best_model = trained_models[np.argmax(scores)].module
    torch.save(best_model.state_dict(), os.path.join(train_params_obj['tuned_model_path'],f"[HPARAM_TUNING]{trial.number}trial.ckpt"))

    return np.mean(scores)

def perf_eval(model_name, trues, preds, configs=None, pred_type='regression'):
    if pred_type == 'classification_binary':
        metric_list = ['Accuracy', 'Recall', 'Precision', 'F1 score', 'Specificity', 'NPV']
        # eval_res = pd.DataFrame(columns=metric_list)
        preds_binary = preds.argmax(1)

        conf_mat = confusion_matrix(trues, preds_binary)
        tn, fp, fn, tp = conf_mat.ravel()

        acc = (tn+tp)/conf_mat.sum()
        rec = tp/(tp+fn)
        pre = tp/(tp+fp)
        spe = tn/(tn+fp)
        npv = tn/(tn+fn)
        f1 = 2*((pre*rec)/(pre+rec))

        eval_res = pd.DataFrame([[model_name,len(trues),acc,rec,pre,f1,spe,npv]], columns=['Model','Support']+metric_list).set_index('Model').apply(np.round, axis=0, decimals=4)

        return (eval_res, conf_mat)
    elif pred_type == 'regression':
        metric_list = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
        # eval_res = pd.DataFrame(columns=metric_list)

        mae = mean_absolute_error(trues, preds)
        mape = np.mean(abs(trues-preds)/trues)
        mse = mean_squared_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        r2 = r2_score(trues, preds)

        eval_res = pd.DataFrame([[model_name, len(trues), mae, mape, mse, rmse, r2]], columns=['Model', 'Support']+metric_list).set_index('Model').apply(np.round, axis=0, decimals=4)

        return eval_res
    elif pred_type == 'generative':
        # cols = ['Origin SEQ', 'Generated SEQ']
        # eval_res = pd.DataFrame(columns=cols)

        assert configs is not None, "Configuration is needed to evaulate Generative model"
        trues_class = pd.Series(token2class(trues.tolist(), configs=configs))
        if trues.shape != preds.shape:
            preds = preds.argmax(1)
        preds_class = pd.Series(token2class(preds.tolist(), configs=configs))

        eval_res = pd.concat([trues_class, preds_class], axis=1)
        eval_res.columns = ['Origin SEQ', 'Generated SEQ']

        return eval_res
    else:
        print(f"Not implemented for {pred_type} type.")
        return
