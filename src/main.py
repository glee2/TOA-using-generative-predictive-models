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
from collections import OrderedDict

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
from model import Encoder_SEQ, Decoder_SEQ, Attention, AttnDecoder_SEQ, SEQ2SEQ
from train_utils import run_epoch, EarlyStopping, perf_eval
from utils import token2class

TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_PAD = '<PAD>'

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
parser.add_argument('--bidirec', default=False, action='store_true')

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
    bidirec = args.bidirec
    n_directions = 2 if bidirec else 1

    if bidirec:
        hidden_dim_enc = hidden_dim * n_directions
    else:
        hidden_dim_enc = hidden_dim

    target_ipc = args.target_ipc
    use_early_stopping = False if args.no_early_stopping else True
    if use_early_stopping: early_stop_patience = int(0.3*max_epochs)

    train_param_name = f"TRAIN_{args.data_type}{n_folds}folds_{learning_rate}lr_{batch_size}batch_{max_epochs}ep"
    best_model_path = os.path.join(root_dir, "models", f"[CV_best_model][{target_ipc}]{train_param_name}.ckpt")

    train_params = {'target_ipc': target_ipc}

    # Sampling for cross validation
    print("Load dataset...")
    tstart = time.time()
    tech_dataset = TechDataset(device=device, data_dir=data_dir, do_transform=False, params=train_params)
    data_loader = DataLoader(tech_dataset, batch_size=batch_size)
    tend = time.time()
    print(f"{np.round(tend-tstart,4)} sec elapsed for loading patents for class [{train_params['target_ipc']}]")

    if args.train:
        sampler = CVSampler(tech_dataset, n_folds=n_folds, test_ratio=0.3)
        cv_idx = sampler.get_idx_dict()
        print(f"#Samples\nTrain: {len(cv_idx[0]['train'])}, Validation: {len(cv_idx[0]['val'])}, Test: {len(cv_idx[0]['test'])}")

        # Ignoring padding index
        padding_idx = tech_dataset.vocab_w2i[TOKEN_PAD]
        # loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

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

            enc = Encoder_SEQ(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=tech_dataset.vocab_size, n_layers=n_layers, bidirec=bidirec,  device=device).to(device=device, dtype=torch.float)

            att = Attention(hidden_dim_enc, hidden_dim)

            dec = AttnDecoder_SEQ(embedding_dim=embedding_dim, vocab_size=tech_dataset.vocab_size, hidden_dim=hidden_dim, hidden_dim_enc=hidden_dim_enc, attention=att, n_layers=n_layers, device=device, max_len=tech_dataset.seq_len).to(device=device, dtype=torch.float)

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

        best_model = trained_models[np.argmax(list(losses_per_fold.values()))].module # best model in CV # unwrap data parallel
        trues_test = Y_test.cpu().detach().numpy()
        outputs_test, z_test = best_model(X_test)
        preds_test = outputs_test.cpu().detach().numpy()
        torch.save(best_model.state_dict(), best_model_path)

        # Qaultitative evaluation
        X_test_class = pd.Series(token2class(X_test.tolist(), vocabulary=tech_dataset.vocab_i2w))

        preds_test, h_test = best_model(X_test.to(device))
        preds_test = preds_test.argmax(1)
        preds_test_class = pd.Series(token2class(preds_test.tolist(), vocabulary=tech_dataset.vocab_i2w))

        test_res = pd.concat([X_test_class, preds_test_class], axis=1)
        test_res.columns = ['TRUE', 'PRED']
        print(test_res)

    else:
        enc = Encoder_SEQ(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=tech_dataset.vocab_size, n_layers=n_layers, bidirec=bidirec, device=device).to(device=device, dtype=torch.float)

        att = Attention(hidden_dim_enc, hidden_dim)

        dec = AttnDecoder_SEQ(embedding_dim=embedding_dim, vocab_size=tech_dataset.vocab_size, hidden_dim=hidden_dim, hidden_dim_enc=hidden_dim_enc, attention=att, n_layers=n_layers, device=device, max_len=tech_dataset.seq_len).to(device=device, dtype=torch.float)

        best_model = SEQ2SEQ(device=device, dataset=tech_dataset, enc=enc, dec=dec, max_len=tech_dataset.seq_len).to(device=device, dtype=torch.float)
        best_model = torch.nn.DataParallel(best_model, device_ids=device_ids)

        best_states = torch.load(best_model_path)
        converted_states = OrderedDict()
        for k, v in best_states.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            converted_states[k] = v

        best_model.load_state_dict(converted_states)

        for batch, (X, Y) in enumerate(data_loader):
            X_class = pd.Series(token2class(X.tolist(), vocabulary=tech_dataset.vocab_i2w))

            preds, h = best_model(X.to(device))
            preds = preds.argmax(1)
            preds_class = pd.Series(token2class(preds.tolist(), vocabulary=tech_dataset.vocab_i2w))

            test_res = pd.concat([X_class, preds_class], axis=1)
            test_res.columns = ['TRUE', 'PRED']
            print(test_res)

            if batch == 10: break

    ## TEST: new sample generation from latent vector only
    xs, ys = next(iter(data_loader))
    x = xs[0].unsqueeze(0).to(device)
    print(f"Generation test\nExample: {token2class(x.tolist(), vocabulary=tech_dataset.vocab_i2w)}")
    o_enc, h_enc = enc(x)
    # h_enc = h_enc.view(n_layers, n_directions, batch_size, hidden_dim)

    # Take the last layer hidden vector as latent vector
    # z = z[-1].view(1, batch_size, -1) # last layer hidden_vector -> (1, batch_size, hidden_dim * n_directions)
    z = h_enc
    new_outputs = torch.zeros(1, tech_dataset.vocab_size, tech_dataset.seq_len)
    next_input = torch.from_numpy(np.tile([tech_dataset.vocab_w2i[TOKEN_SOS]], 1)).to(device)
    for t in range(1, tech_dataset.seq_len):
        embedded = dec.dropout(dec.embedding(next_input.unsqueeze(1)))
        gru_input = torch.cat((embedded, z[-1].unsqueeze(0)), dim=2) # Replace attention weights with latent vector
        o_dec, h_dec = dec.gru(gru_input, z)
        output = dec.fc_out(torch.cat((o_dec.squeeze(1), z[-1], embedded.squeeze(1)), dim=1))
        prediction = output.argmax(1)

        next_input = prediction
        new_outputs[:,:,t] = output
    new_outputs = new_outputs.argmax(1)

    print(f"Generated output (using the original latent vector from encoder): {token2class(new_outputs.tolist(), vocabulary=tech_dataset.vocab_i2w)}")

    # What if adding noise to latent vector?
    new_z = z + z.mean().item() * 1e-2
    new_outputs = torch.zeros(1, tech_dataset.vocab_size, tech_dataset.seq_len)
    next_input = torch.from_numpy(np.tile([tech_dataset.vocab_w2i[TOKEN_SOS]], 1)).to(device)
    for t in range(1, tech_dataset.seq_len):
        embedded = dec.dropout(dec.embedding(next_input.unsqueeze(1)))
        gru_input = torch.cat((embedded, new_z[-1].unsqueeze(0)), dim=2) # Replace attention weights with latent vector
        o_dec, h_dec = dec.gru(gru_input, h_enc)
        output = dec.fc_out(torch.cat((o_dec.squeeze(1), z[-1], embedded.squeeze(1)), dim=1))
        prediction = output.argmax(1)

        next_input = prediction
        new_outputs[:,:,t] = output
    new_outputs = new_outputs.argmax(1)

    print(f"Generated output (using changed latent vector from encoder): {token2class(new_outputs.tolist(), vocabulary=tech_dataset.vocab_i2w)}")
