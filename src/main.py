# Notes
'''
Author: Gyumin Lee
Version: 0.2
Description (primary changes): Add attention decoder
'''

# Set root directory
root_dir = '/home2/glee/Tech_Gen/'
import sys
sys.path.append(root_dir)

import copy
import gc
import os
import argparse
import math
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
sys.path.append("/share/tml_package")
from tml import utils
from scipy import io
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.nn import DataParallel as DP
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from data import TechDataset, CVSampler
from model import Encoder_SEQ, Decoder_SEQ, AttnDecoder_SEQ, SEQ2SEQ
from train_utils import run_epoch, EarlyStopping, perf_eval

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', default='sequence')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--n_folds', default=1, type=int)
parser.add_argument('--learning_rate', default=5e-3, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--max_epochs', default=2, type=int)
parser.add_argument('--n_gpus', default=4, type=int)
parser.add_argument('--embedding_dim', default=32, type=int)
parser.add_argument('--hidden_dim', default=32, type=int)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--no_early_stopping', dest='no_early_stopping', action='store_true')
parser.add_argument('--target_ipc', default='A61C', type=str)

if __name__=="__main__":
    args = parser.parse_args()

    data_dir = os.path.join(root_dir, 'data')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        device_ids = np.argsort(list(map(torch.cuda.memory_allocated, device_ids)))[:args.n_gpus]
        device_ids = list(map(lambda x: torch.device('cuda', x),list(device_ids)))
    else:
        device = torch.device('cpu')
        device_ids = []

    # Set hyperparameters
    n_folds = args.n_folds
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers

    target_ipc = args.target_ipc
    use_early_stopping = False if args.no_early_stopping else True
    if use_early_stopping: early_stop_patience = int(0.3*max_epochs)

    train_param_name = f"TRAIN_{args.data_type}{n_folds}folds_{learning_rate}lr_{batch_size}batch_{max_epochs}ep"
    best_model_path = os.path.join(root_dir, "models", "[CV_best_model]"+train_param_name+".ckpt")

    train_params = {'target_ipc': target_ipc}

    if args.train:
        # Sampling for cross validation
        print("Load dataset...")
        tstart = time.time()
        tech_dataset = TechDataset(device=device, data_dir=data_dir, do_transform=False, params=train_params)
        data_loader = DataLoader(tech_dataset, batch_size=batch_size)
        tend = time.time()
        print(f"{np.round(tend-tstart,4)} sec elapsed for loading patents for class [{train_params['target_ipc']}]")

        sampler = CVSampler(tech_dataset, n_folds=n_folds, test_ratio=0.3)
        cv_idx = sampler.get_idx_dict()
        print(f"#Samples\nTrain: {len(cv_idx[0]['train'])}, Validation: {len(cv_idx[0]['val'])}, Test: {len(cv_idx[0]['test'])}")

        loss_fn = torch.nn.CrossEntropyLoss()

        # Leave test set
        test_dataset = Subset(tech_dataset, cv_idx[0]['test'])
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=8)
        X_test, Y_test = next(iter(test_loader))
        X_test, Y_test = X_test.to(device, dtype=torch.long), Y_test.to(device, dtype=torch.long)

        trues_cv, preds_cv = [], []
        trained_models, losses_per_fold = {}, {}
        for fold in tqdm(range(n_folds)):
            train_dataset = Subset(tech_dataset, cv_idx[fold]['train'])
            val_dataset = Subset(tech_dataset, cv_idx[fold]['val'])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
            val_loader_cv = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=8)

            enc = Encoder_SEQ(device=device, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=tech_dataset.vocab_size, n_layers=n_layers).to(device=device, dtype=torch.float)
            # dec = Decoder_SEQ(device=device, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=tech_dataset.vocab_size, n_layers=n_layers).to(device=device, dtype=torch.float)
            dec = AttnDecoder_SEQ(device=device, embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=tech_dataset.vocab_size, n_layers=n_layers, max_len=tech_dataset.seq_len).to(device=device, dtype=torch.float)
            model = SEQ2SEQ(device=device, dataset=tech_dataset, enc=enc, dec=dec, max_len=tech_dataset.seq_len).to(device=device, dtype=torch.float)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            train_losses, val_losses = [], []
            if use_early_stopping:
                early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path="../models/ES_checkpoint_"+train_param_name+".ckpt")
            for ep in range(max_epochs):
                print(f"Epoch {ep+1}\n"+str("-"*25))
                train_loss = run_epoch(train_loader, model, loss_fn, mode='train', optimizer=optimizer, device=device)
                val_loss = run_epoch(val_loader, model, loss_fn, mode='test', device=device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Avg train loss: {train_loss:>5f}, Avg val loss: {val_loss:>5f}\n")

                if use_early_stopping:
                    model = early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopped\n")
                        break

            trained_models[fold] = model
            losses_per_fold[fold] = np.average(val_losses)

            X_val_cv, Y_val_cv = next(iter(val_loader_cv))
            trues_cv.append(Y_val_cv.cpu().detach().numpy())
            outputs_cv, z_cv = model.module(X_val_cv.to(device=device)) # unwrap data parallel
            preds_cv.append(outputs_cv.cpu().detach().numpy())

        trues_cv = np.concatenate(trues_cv)
        preds_cv = np.concatenate(preds_cv)
        # eval_cv, conf_cv = perf_eval('CNN_cv', trues_cv, preds_cv)
        # conf_cv = conf_cv.astype(np.int32)

        best_model = trained_models[np.argmax(list(losses_per_fold.values()))].module # best model in CV # unwrap data parallel
        trues_test = Y_test.cpu().detach().numpy()
        outputs_test, z_test = best_model(X_test)
        preds_test = outputs_test.cpu().detach().numpy()
        # eval_test, conf_test = perf_eval('CNN_test', trues_test, preds_test)
        # eval_res = pd.concat([eval_cv, eval_test], axis=0)
        # eval_res.to_csv("../results/[Eval_res]"+train_param_name+".csv")
        # np.savetxt("../results/[Confmat_CV]"+train_param_name+".txt", conf_cv, delimiter=',', fmt="%d")
        # np.savetxt("../results/[Confmat_TEST]"+train_param_name+".txt", conf_test, delimiter=',', fmt="%d")
        torch.save(best_model.state_dict(), best_model_path)

    else:
        best_model = SEQ2SEQ(device=device, dataset=tech_dataset, enc=enc, dec=dec, max_len=tech_dataset.seq_len).to(device=device, dtype=torch.float)
        best_model = torch.nn.DataParallel(best_model, device_ids=device_ids)
        best_model = best_model.load_state_dict(torch.load(best_model_path))
