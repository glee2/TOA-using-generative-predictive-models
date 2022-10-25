# Notes
'''
Author: Gyumin Lee
Version: 0.4
Description (primary changes): Add functions for hyperparameter tuning
'''

# Set root directory
root_dir = '/home2/glee/Tech_Gen/'
import sys
sys.path.append(root_dir)

import os
import copy
import functools
import operator
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

from data import TechDataset, CVSampler
from model import Encoder_SEQ, Attention, AttnDecoder_SEQ, SEQ2SEQ, Predictor
from utils import token2class

from data import TechDataset, CVSampler
from model import Encoder_SEQ, Attention, AttnDecoder_SEQ, SEQ2SEQ, Predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=10, verbose=True, delta=0, path='../models/checkpoint.ckpt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
            path (str): checkpoint저장 경로
        """
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
        '''validation loss가 감소하면 모델을 저장한다.'''
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

def build_model(model_params={}, trial=None):
    device = model_params['device']
    device_ids = model_params['device_ids']

    if trial is not None:
        model_params['n_layers'] = trial.suggest_int("n_layers", 1, 3)
        model_params['embedding_dim'] = trial.suggest_categorical("embedding_dim", [32, 64, 128, 256])
        model_params['hidden_dim'] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
        model_params['latent_dim'] = trial.suggest_int("latent_dim", model_params['hidden_dim'] * model_params['n_layers'] * model_params['n_directions'], model_params['hidden_dim'] * model_params['n_layers'] * model_params['n_directions'])

    ## Construct networks
    enc = Encoder_SEQ(params=model_params).to(device=device, dtype=torch.float)
    att = Attention(params=model_params).to(device=device, dtype=torch.float)
    dec = AttnDecoder_SEQ(attention=att, params=model_params).to(device=device, dtype=torch.float)
    pred = Predictor(params=model_params).to(device=device, dtype=torch.float)
    model = SEQ2SEQ(device=device, enc=enc, dec=dec, pred=pred, vocab=model_params['vocabulary'], max_len=model_params['max_len']).to(device=device, dtype=torch.float)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    return model

def train_model(model, train_loader, val_loader, model_params={}, train_params={}, trial=None):
    device = model_params['device']

    ## Loss function and optimizers
    loss_recon = torch.nn.CrossEntropyLoss(ignore_index=model_params['padding_idx'])
    loss_y = torch.nn.MSELoss()

    if trial is not None:
        train_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        max_epochs = train_params['max_epochs_for_tune']
        early_stop_patience = train_params['early_stop_patience_for_tune']
    else:
        max_epochs = train_params['max_epochs']
        early_stop_patience = train_params['early_stop_patience']

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    ## Training
    if train_params['use_early_stopping']:
        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=os.path.join(train_params['root_dir'],"models/ES_checkpoint_"+train_params['train_param_name']+".ckpt"))
    for ep in range(max_epochs):
        print(f"Epoch {ep+1}\n"+str("-"*25))
        train_loss = run_epoch(train_loader, model, loss_recon, loss_y, mode='train', optimizer=optimizer, loss_weights=train_params['loss_weights'], device=device)
        val_loss = run_epoch(val_loader, model, loss_recon, loss_y, mode='test', loss_weights=train_params['loss_weights'], device=device)
        print(f"Avg train loss: {train_loss['total']:>5f}, Avg val loss: {val_loss['total']:>5f}\n")

        if train_params['use_early_stopping']:
            model = early_stopping(val_loss['total'], model)
            if early_stopping.early_stop:
                print("Early stopped\n")
                break
    return model

def validate_model(model, val_loader, model_params={}):
    trues_recon_val, trues_y_val = [], []
    preds_recon_val, preds_y_val = [], []
    for batch, (X_batch, Y_batch) in enumerate(val_loader):
        trues_recon_val.append(X_batch.cpu().detach().numpy())
        trues_y_val.append(Y_batch.cpu().detach().numpy())
        preds_recon_batch, preds_y_batch, z_batch = model.module(X_batch.to(device=model_params['device']))
        preds_recon_val.append(preds_recon_batch.cpu().detach().numpy())
        preds_y_val.append(preds_y_batch.cpu().detach().numpy())
    return [np.concatenate(x) for x in [trues_recon_val, trues_y_val, preds_recon_val, preds_y_val]]

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
        trues_recon_val, trues_y_val, preds_recon_val, preds_y_val = validate_model(model, val_loader, model_params=model_params_obj)

        score_mse = mean_squared_error(trues_y_val, preds_y_val)

        scores.append(score_mse)
        trained_models.append(model)

    best_model = trained_models[np.argmax(scores)].module
    torch.save(best_model.state_dict(), os.path.join(train_params_obj['model_path'],"hparam_tuning",f"[HPARAM_TUNING]{trial.number}trial.ckpt"))

    return np.mean(scores)

def perf_eval(model_name, trues, preds, pred_type='regression', vocabulary=None):
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

        assert vocabulary is not None, "Vocabulary is needed to evaulate Generative model"
        trues_class = pd.Series(token2class(trues.tolist(), vocabulary=vocabulary))
        if trues.shape != preds.shape:
            preds = preds.argmax(1)
        preds_class = pd.Series(token2class(preds.tolist(), vocabulary=vocabulary))

        eval_res = pd.concat([trues_class, preds_class], axis=1)
        eval_res.columns = ['Origin SEQ', 'Generated SEQ']

        return eval_res
    else:
        print(f"Not implemented for {pred_type} type.")
        return
