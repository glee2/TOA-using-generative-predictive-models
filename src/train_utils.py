# Notes
'''
Author: Gyumin Lee
Version: 0.3
Description (primary changes): Add loss_weights
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

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

    batch_losses = []
    if mode == 'train':
        model.train()
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device, dtype=torch.long), Y.to(device, dtype=torch.float) # X: (batch_size, seq_len)
            preds_recon, preds_y, z = model(X) # preds_recon: (batch_size, vocab_size, seq_len), preds_y: (batch_size, 1), z: (n_layers, batch_size, hidden_dim * n_directions)
            trues_recon = X.clone()
            trues_y = Y.clone()
            # print(f"Recon loss: {loss_recon(preds_recon, trues_recon)}, Y loss: {loss_y(preds_y, trues_y)}")
            # loss = loss_recon(preds_recon, trues_recon) + loss_y(preds_y, trues_y)*10
            loss = sum([loss_weights['recon']*loss_recon(preds_recon, trues_recon), loss_weights['y']*loss_y(preds_y, trues_y)])
            batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            if batch % 10 == 0 or batch == len(dataloader)-1:
                loss, current = loss.item(), batch*len(X)
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
                # loss = loss_recon(preds_recon, trues_recon) + loss_y(preds_y, trues_y)*10
                # loss = sum([loss_recon(preds_recon, trues_recon), loss_y(preds_y, trues_y)*5])
                loss = sum([loss_weights['recon']*loss_recon(preds_recon, trues_recon), loss_weights['y']*loss_y(preds_y, trues_y)])
                test_loss += loss.item()
                batch_losses.append(loss.item())
        test_loss /= n_batches
        # print(f"Avg loss: {test_loss:>8f}\n")

    return np.average(batch_losses)

def perf_eval(model_name, trues, preds):
    metric_list = ['Accuracy', 'Recall', 'Precision', 'F1 score', 'Specificity', 'NPV']
    eval_res = pd.DataFrame(columns=metric_list)
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

    return eval_res, conf_mat
