# Notes
'''
Author: Gyumin Lee
Version: 0.3
Description (primary changes): Add ipc_level, Add loss_weights
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

import optuna

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from data import TechDataset, CVSampler
from model import Encoder_SEQ, Attention, AttnDecoder_SEQ, SEQ2SEQ, Predictor
from train_utils import run_epoch, EarlyStopping, perf_eval, objective_cv, build_model, train_model, validate_model
from utils import token2class

TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_PAD = '<PAD>'

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', default='sequence')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--tune', default=False, action='store_true')
parser.add_argument('--n_trials', default=2, type=int)
parser.add_argument('--n_folds', default=1, type=int)
parser.add_argument('--learning_rate', default=5e-3, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epochs', default=2, type=int)
parser.add_argument('--n_gpus', default=4, type=int)
parser.add_argument('--embedding_dim', default=32, type=int)
parser.add_argument('--hidden_dim', default=32, type=int)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--no_early_stopping', dest='no_early_stopping', action='store_true')
parser.add_argument('--target_ipc', default='A23L', type=str)
parser.add_argument('--ipc_level', default=3, type=int, help="IPC level. 1: Section-Class, 2: Section-Class-Sub Class, 3: Section-Class-Sub Class-Group(main or sub)")
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

    ## Set hyperparameters for model training (FIXED)
    bidirec = args.bidirec
    n_directions = 2 if bidirec else 1
    output_dim_predictor = 1
    n_folds = args.n_folds
    max_epochs = args.max_epochs
    dropout = 0.5
    loss_weights = {'recon': 2, 'y': 8}
    model_path = os.path.join(root_dir, "models")

    ## Set hyperparameters for model training (To be TUNED)
    if args.train and args.tune:
        n_layers = args.n_layers = None
        embedding_dim = args.embedding_dim = None
        hidden_dim = args.hidden_dim = None
        latent_dim = None
        learning_rate = args.learning_rate = None
        batch_size = args.batch_size = None

        train_param_name = "HPARAM_TUNING"
        final_model_path = None
    else:
        n_layers = args.n_layers
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        latent_dim = hidden_dim*n_layers*n_directions
        learning_rate = args.learning_rate
        batch_size = args.batch_size

        train_param_name = f"{n_layers}layers_{embedding_dim}emb_{hidden_dim}hid_{np.round(learning_rate,4)}lr_{batch_size}batch_{max_epochs}ep"
        final_model_path = os.path.join(model_path, f"[Final_model][{args.target_ipc}]{train_param_name}.ckpt")

    use_early_stopping = False if args.no_early_stopping else True
    if use_early_stopping: early_stop_patience = int(0.3*max_epochs)

    train_params = copy.deepcopy(vars(args))
    train_params.pop('train') # To avoid conflict
    train_params.update({'root_dir': root_dir,
                         'loss_weights': loss_weights,
                         'use_early_stopping': use_early_stopping,
                         'early_stop_patience': early_stop_patience,
                         'train_param_name': train_param_name,
                         'model_path': model_path,
                         'max_epochs_for_tune': 50,
                         'early_stop_patience_for_tune': 20})
    model_params = copy.deepcopy(vars(args))
    model_params.pop('train') # To avoid conflict
    model_params.update({'device': device,
                         'device_ids': device_ids,
                         'n_directions': n_directions,
                         'latent_dim': latent_dim,
                         'output_dim_predictor': output_dim_predictor,
                         'dropout': dropout})

    # Sampling for cross validation
    print("Load dataset...")
    tstart = time.time()
    tech_dataset = TechDataset(device=device, data_dir=data_dir, do_transform=False, params=train_params)
    tend = time.time()
    print(f"{np.round(tend-tstart,4)} sec elapsed for loading patents for class [{train_params['target_ipc']}]")

    model_params.update({'vocabulary': tech_dataset.vocab_w2i,
                         'vocabulary_rev': tech_dataset.vocab_i2w,
                         'padding_idx': tech_dataset.vocab_w2i[TOKEN_PAD],
                         'vocab_size': tech_dataset.vocab_size,
                         'max_len': tech_dataset.seq_len})

    if args.train:
        sampler = CVSampler(tech_dataset, n_folds=n_folds, test_ratio=0.3)
        cv_idx = sampler.get_idx_dict()
        print(f"#Samples\nTrain: {len(cv_idx[0]['train'])}, Validation: {len(cv_idx[0]['val'])}, Test: {len(cv_idx[0]['test'])}")

        if args.tune:
            optuna_obj = lambda trial: objective_cv(trial, dataset=tech_dataset, cv_idx=cv_idx, model_params=model_params, train_params=train_params)

            study = optuna.create_study(direction='minimize')
            study.optimize(optuna_obj, n_trials=args.n_trials, timeout=600)
            best_params = study.best_trial.params

            print(f"Best trial:\n  MSE: {study.best_trial.value}\n  Params:")
            for k, v in best_params.items():
                print(f"    {k}: {v}")

            train_params.update({k: v for k,v in best_params.items() if k in train_params.keys()})
            model_params.update({k: v for k,v in best_params.items() if k in model_params.keys()})

            train_param_name = f"{model_params['n_layers']}layers_{model_params['embedding_dim']}emb_{model_params['hidden_dim']}hid_{np.round(train_params['learning_rate'],4)}lr_{train_params['batch_size']}batch_{train_params['max_epochs']}ep"
            final_model_path = os.path.join(model_path, f"[Final_model][{args.target_ipc}]{train_param_name}.ckpt")

        ## Construct datasets
        train_idx = cv_idx[0]['train']
        val_idx = cv_idx[0]['val']
        test_idx = cv_idx[0]['test']
        whole_idx = np.concatenate([train_idx, val_idx])

        train_dataset = Subset(tech_dataset, train_idx)
        val_dataset = Subset(tech_dataset, val_idx)
        test_dataset = Subset(tech_dataset, test_idx)
        whole_dataset = Subset(tech_dataset, whole_idx)

        train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=4)
        whole_loader = DataLoader(whole_dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=4)

        ## Load best model
        final_model = build_model(model_params)
        if args.tune:
            best_states = torch.load(f"{train_params['model_path']}_{study.best_trial.number}trial.ckpt")
            converted_states = OrderedDict()
            for k, v in best_states.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                converted_states[k] = v
            final_model.load_state_dict(converted_states)
        else:
            final_model = train_model(final_model, train_loader, val_loader, model_params, train_params)
        torch.save(final_model.state_dict(), final_model_path) # Finalize

        ## Evaluation on train dataset
        trues_recon_train, trues_y_train, preds_recon_train, preds_y_train = validate_model(final_model, whole_loader, model_params)
        eval_recon_train = perf_eval("TRAIN_SET", trues_recon_train, preds_recon_train, pred_type='generative', vocabulary=model_params['vocabulary_rev'])
        eval_y_train = perf_eval("TRAIN_SET", trues_y_train, preds_y_train, pred_type='regression')

        ## Evaluation on test dataset
        trues_recon_test, trues_y_test, preds_recon_test, preds_y_test = validate_model(final_model, test_loader, model_params)
        eval_recon_test = perf_eval("TEST_SET", trues_recon_test, preds_recon_test, pred_type='generative', vocabulary=model_params['vocabulary_rev'])
        eval_y_test = perf_eval("TEST_SET", trues_y_test, preds_y_test, pred_type='regression')

        eval_y_res = pd.concat([eval_y_train, eval_y_test], axis=0)

        result_path = os.path.join(root_dir, "results")
        with pd.ExcelWriter(os.path.join(result_path,f"[RESULT][{args.target_ipc}]{train_param_name}.xlsx")) as writer:
            eval_y_res.to_excel(writer, sheet_name="Regression")
            eval_recon_train.to_excel(writer, sheet_name="Generative_TRAIN")
            eval_recon_test.to_excel(writer, sheet_name="Generative_TEST")
    else:
        final_model = build_model(model_params)
        best_states = torch.load(final_model_path)
        converted_states = OrderedDict()
        for k, v in best_states.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            converted_states[k] = v
        final_model.load_state_dict(converted_states)

        data_loader = DataLoader(tech_dataset, batch_size=args.batch_size)

        trues_recon, trues_y, preds_recon, preds_y = validate_model(final_model, data_loader, model_params)
        eval_recon = perf_eval("LOADED_MODEL", trues_recon, preds_recon, pred_type='generative', vocabulary=model_params['vocabulary_rev'])
        eval_y = perf_eval("LOADED_MODEL", trues_y, preds_y, pred_type='regression')

    ## TEST: new sample generation from latent vector only
    data_loader = DataLoader(tech_dataset, batch_size=10)
    xs, ys = next(iter(data_loader))
    x = xs[0].unsqueeze(0).to(device)
    print(f"Generation test\nExample: {token2class(x.tolist(), vocabulary=model_params['vocabulary_rev'])}")
    o_enc, h_enc = final_model.module.encoder(x)
    # h_enc = h_enc.view(n_layers, n_directions, batch_size, hidden_dim)

    # Take the last layer hidden vector as latent vector
    # z = z[-1].view(1, batch_size, -1) # last layer hidden_vector -> (1, batch_size, hidden_dim * n_directions)
    z = h_enc
    new_outputs = torch.zeros(1, model_params['vocab_size'], model_params['max_len'])
    next_input = torch.from_numpy(np.tile([model_params['vocabulary'][TOKEN_SOS]], 1)).to(device)
    for t in range(1, model_params['max_len']):
        embedded = final_model.module.decoder.dropout(final_model.module.decoder.embedding(next_input.unsqueeze(1)))
        gru_input = torch.cat((embedded, z[-1].unsqueeze(0)), dim=2) # Replace attention weights with latent vector
        o_dec, h_dec = final_model.module.decoder.gru(gru_input, z)
        output = final_model.module.decoder.fc_out(torch.cat((o_dec.squeeze(1), z[-1], embedded.squeeze(1)), dim=1))
        prediction = output.argmax(1)

        next_input = prediction
        new_outputs[:,:,t] = output
    new_outputs = new_outputs.argmax(1)

    print(f"Generated output (using the original latent vector from encoder): {token2class(new_outputs.tolist(), vocabulary=tech_dataset.vocab_i2w)}")

    # What if adding noise to latent vector?
    # new_z = z + z.mean().item() * 5e-1
    new_z = z + torch.rand((z.shape)).to(device) * 1e-4
    new_outputs = torch.zeros(1, model_params['vocab_size'], model_params['max_len'])
    next_input = torch.from_numpy(np.tile([model_params['vocabulary'][TOKEN_SOS]], 1)).to(device)
    for t in range(1, model_params['max_len']):
        embedded = final_model.module.decoder.dropout(final_model.module.decoder.embedding(next_input.unsqueeze(1)))
        gru_input = torch.cat((embedded, new_z[-1].unsqueeze(0)), dim=2) # Replace attention weights with latent vector
        o_dec, h_dec = final_model.module.decoder.gru(gru_input, h_enc)
        output = final_model.module.decoder.fc_out(torch.cat((o_dec.squeeze(1), z[-1], embedded.squeeze(1)), dim=1))
        prediction = output.argmax(1)

        next_input = prediction
        new_outputs[:,:,t] = output
    new_outputs = new_outputs.argmax(1)

    print(f"Generated output (using changed latent vector from encoder): {token2class(new_outputs.tolist(), vocabulary=tech_dataset.vocab_i2w)}")
