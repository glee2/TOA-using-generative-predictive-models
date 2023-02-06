# Notes
'''
Author: Gyumin Lee
Version: 0.7
Description (primary changes): Employ "Accelerator" from huggingface, to automate gpu-distributed training
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
import pickle
import re
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
from accelerate import Accelerator
import pytorch_model_summary

import optuna
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration import SkoptSampler

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from data import TechDataset, CVSampler
from models import Transformer
from train_utils import run_epoch, EarlyStopping, perf_eval, objective_cv, build_model, train_model, validate_model
from utils import token2class, DotDict

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str)
parser.add_argument("--target_ipc", type=str)
parser.add_argument("--do_save", default=None, action="store_true")
parser.add_argument("--pred_type", type=str)
parser.add_argument("--do_train", default=None, action="store_true")
parser.add_argument("--do_tune", default=None, action="store_true")
parser.add_argument("--n_folds", type=int)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--use_accelerator", default=None, action="store_true")

if __name__=="__main__":
    ''' PART 1: Configuration '''
    configs = DotDict().load("configs.json")
    org_config_keys = {key: list(configs[key].keys()) for key in configs.keys()}

    args = parser.parse_args()
    instant_configs = {key: value for (key, value) in vars(args).items() if value is not None} # if any argument passed when main.py executed
    instant_configs_for_update = {configkey: {key: value for (key,value) in instant_configs.items() if key in org_config_keys[configkey]} for configkey in org_config_keys.keys()}
    for key, value in configs.items():
        value.update(instant_configs_for_update[key])

    data_dir = os.path.join(root_dir, "data")
    model_dir = os.path.join(root_dir, "models")

    if configs.train.use_accelerator:
        accelerator = Accelerator()
        device_ids = list(range(torch.cuda.device_count()))
        device = accelerator.device

        configs.train.update({"accelerator": accelerator})
        # configs.model.update({"use_accelerator": configs.train.use_accelerator})
    else:
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            gpu_usages = [np.sum([float(usage.split("uses")[-1].replace(" ","").replace("MB","")) for usage in torch.cuda.list_gpu_processes(id).split("GPU memory") if not usage=="" and "no processes are running" not in usage]) for id in device_ids]
            device_ids = np.argsort(gpu_usages)[:configs.train.n_gpus]
            device_ids = list(map(lambda x: torch.device('cuda', x),list(device_ids)))
            device = device_ids[0] # main device
        else:
            device = torch.device('cpu')
            device_ids = []

    configs.data.update({"root_dir": root_dir,
                            "data_dir": data_dir,
                            "model_dir": model_dir})
    configs.train.update({"device": device,
                            "device_ids": device_ids,
                            "root_dir": root_dir,
                            "data_dir": data_dir,
                            "model_dir": model_dir,
                            "early_stop_patience": int(0.3*configs.train.max_epochs)})
    configs.model.update({"device": device,
                            "device_ids": device_ids,
                            "n_directions": 2 if configs.model.bidirec else 1,
                            "n_outputs": 1 if configs.data.pred_type=="regression" else 2,
                            "use_accelerator": configs.train.use_accelerator})

    ## Set hyperparameters for model training (To be TUNED)
    if configs.train.do_train and configs.train.do_tune:
        n_layers = configs.model.n_layers = None
        d_embedding = configs.model.d_embedding = None
        d_hidden = configs.model.d_hidden = None
        d_latent = None
        learning_rate = configs.train.learning_rate = None
        batch_size = configs.train.batch_size = None
        config_name = "HPARAM_TUNING"
        final_model_path = None
    else:
        n_layers = configs.model.n_layers
        d_embedding = configs.model.d_embedding
        d_hidden = configs.model.d_hidden
        d_latent = configs.model.n_layers * configs.model.d_hidden * configs.model.n_directions
        config_name = f"{n_layers}layers_{d_embedding}emb_{d_hidden}hid_{configs.model.n_directions}direc_{np.round(configs.train.learning_rate,4)}lr_{configs.train.batch_size}batch_{configs.train.max_epochs}ep"
        final_model_path = os.path.join(model_dir, f"[Final_model][{configs.data.target_ipc}]{config_name}.ckpt")

    configs.model.update({"d_latent": d_latent})
    configs.train.update({"config_name": config_name,
                            "final_model_path": final_model_path})

    ''' PART 2: Dataset setting '''
    print("Load dataset...")
    tstart = time.time()
    dataset_config_name = "-".join([str(key)+"="+str(value) for (key,value) in configs.data.items() if key in org_config_keys["data"]])
    dataset_path = os.path.join(data_dir, "pickled_dataset", "[tech_dataset]"+dataset_config_name+".pickle")
    if os.path.exists(dataset_path) and configs.data.do_save is False:
        with open(dataset_path, "rb") as f:
            tech_dataset = pickle.load(f)   # Load pickled dataset if dataset with same configuration already saved
        print("Pickled dataset loaded")
    else:
        tech_dataset = TechDataset(configs.data)
        with open(dataset_path, "wb") as f:
            tech_dataset.rawdata = None
            pickle.dump(tech_dataset, f)
    tend = time.time()
    print(f"{np.round(tend-tstart,4)} sec elapsed for loading patents for class [{configs.data.target_ipc}]")

    configs.model.update({"vocabulary": tech_dataset.vocab_w2i,
                            "vocabulary_rev": tech_dataset.vocab_i2w,
                            "n_enc_vocab": tech_dataset.vocab_size,
                            "n_dec_vocab": tech_dataset.vocab_size,
                            "n_enc_seq": tech_dataset.seq_len,
                            "n_dec_seq": tech_dataset.seq_len,
                            "i_padding": tech_dataset.vocab_w2i["<PAD>"]})

    ''' PART 3: Training '''
    if configs.train.do_train:
        sampler = CVSampler(tech_dataset, n_folds=configs.train.n_folds, test_ratio=0.3, stratify=True)
        cv_idx = sampler.get_idx_dict()
        print(f"#Samples\nTrain: {len(cv_idx[0]['train'])}, Validation: {len(cv_idx[0]['val'])}, Test: {len(cv_idx[0]['test'])}")

        ''' PART 3-1: Hyperparmeter tuning '''
        if configs.train.do_tune:
            configs.train.update({"tuned_model_path": os.path.join(model_dir,"hparam_tuning")})
            optuna_obj = lambda trial: objective_cv(trial, dataset=tech_dataset, cv_idx=cv_idx, model_params=configs.model, train_params=configs.train)

            opt_sampler = TPESampler()
            study = optuna.create_study(direction='minimize')
            study.optimize(optuna_obj, n_trials=configs.train.n_trials, gc_after_trial=True)
            best_params = study.best_trial.params

            print(f"Best trial:\n  MSE: {study.best_trial.value}\n  Params:")
            for k, v in best_params.items():
                print(f"    {k}: {v}")

            configs.train.update({k: v for k,v in best_params.items() if k in configs.train.keys()})
            configs.model.update({k: v for k,v in best_params.items() if k in configs.model.keys()})
            config_name = f"{configs.model.n_layers}layers_{configs.model.d_embedding}emb_{configs.model.d_hidden}hid_{configs.model.n_directions}direc_{np.round(configs.train.learning_rate, 4)}lr_{configs.train.batch_size}batch_{configs.train.max_epochs}ep"
            final_model_path = os.path.join(model_dir, f"[Final_model][{configs.data.target_ipc}]{config_name}.ckpt")

        ## Construct datasets
        train_idx = cv_idx[0]['train']
        val_idx = cv_idx[0]['val']
        test_idx = cv_idx[0]['test']
        whole_idx = np.concatenate([train_idx, val_idx])

        train_dataset = Subset(tech_dataset, train_idx)
        val_dataset = Subset(tech_dataset, val_idx)
        test_dataset = Subset(tech_dataset, test_idx)
        whole_dataset = Subset(tech_dataset, whole_idx)

        train_loader = DataLoader(train_dataset, batch_size=configs.train.batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=configs.train.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=configs.train.batch_size, shuffle=False, num_workers=4)
        whole_loader = DataLoader(whole_dataset, batch_size=configs.train.batch_size, shuffle=False, num_workers=4)

        ## Load best model
        final_model = build_model(configs.model)
        x_input, _ = next(iter(train_loader))
        if re.search("^1.", torch.__version__) is not None:
            print(pytorch_model_summary.summary(final_model.module, torch.zeros(x_input.shape, device=device, dtype=torch.long), torch.zeros(x_input.shape, device=device, dtype=torch.long), show_input=True, max_depth=None, show_parent_layers=True))
        else:
            print("INFO: pytorch-model-summary does not support PyTorch 2.0, so just print model structure")
            print(final_model)
            torch._dynamo.config.verbose = True
            torch._dynamo.config.suppress_errors = True
        if configs.train.do_tune:
            best_states = torch.load(os.path.join(configs.train.tuned_model_path,f"[HPARAM_TUNING]{study.best_trial.number}trial.ckpt"))
            converted_states = OrderedDict()
            for k, v in best_states.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                converted_states[k] = v
            final_model.load_state_dict(converted_states)
        else:
            final_model = train_model(final_model, train_loader, val_loader, configs.model, configs.train)
        torch.save(final_model.state_dict(), final_model_path) # Finalize

        ## Evaluation on train dataset
        trues_recon_train, preds_recon_train = validate_model(final_model, whole_loader, configs.model)
        # trues_y_train, preds_y_train = validate_model(final_model, whole_loader, configs.model)
        eval_recon_train = perf_eval("TRAIN_SET", trues_recon_train, preds_recon_train, configs=configs, pred_type='generative')
        # eval_y_train = perf_eval("TRAIN_SET", trues_y_train, preds_y_train, pred_type=configs.data.pred_type)
        # if configs.data.pred_type == "classification":
        #     eval_y_train, confmat_y_train = eval_y_train

        ## Evaluation on test dataset
        trues_recon_test, preds_recon_test = validate_model(final_model, test_loader, configs.model)
        # trues_y_test, preds_y_test = validate_model(final_model, test_loader, configs.model)
        eval_recon_test = perf_eval("TEST_SET", trues_recon_test, preds_recon_test, configs=configs,  pred_type='generative')
        # eval_y_test = perf_eval("TEST_SET", trues_y_test, preds_y_test, pred_type=configs.data.pred_type)
        # if configs.data.pred_type == "classification":
        #     eval_y_test, confmat_y_test = eval_y_test

        # eval_y_res = pd.concat([eval_y_train, eval_y_test], axis=0)
        eval_recon_res = pd.concat([eval_recon_train, eval_recon_test], axis=0)

        result_path = os.path.join(root_dir, "results")
        # with pd.ExcelWriter(os.path.join(result_path,f"[RESULT][{args.target_ipc}]{train_param_name}.xlsx")) as writer:
        #     eval_y_res.to_excel(writer, sheet_name=args.pred_type)
            # eval_recon_train.to_excel(writer, sheet_name="Generative_TRAIN")
            # eval_recon_test.to_excel(writer, sheet_name="Generative_TEST")

        print("Training is done!\n")
    else:
        final_model = build_model(configs.model)
        if os.path.exists(final_model_path):
            best_states = torch.load(final_model_path)
        else:
            raise Exception("Model need to be trained first")
        converted_states = OrderedDict()
        for k, v in best_states.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            converted_states[k] = v
        final_model.load_state_dict(converted_states)

        data_loader = DataLoader(tech_dataset, batch_size=configs.train.batch_size)

        trues_y, preds_y = validate_model(final_model, data_loader, configs.model)
        # eval_recon = perf_eval("LOADED_MODEL", trues_recon, preds_recon, pred_type='generative', vocabulary=model_params['vocabulary_rev'])
        eval_y = perf_eval("LOADED_MODEL", trues_y, preds_y, pred_type=configs.data.pred_type)
        if configs.data.pred_type == "classification":
            eval_y, confmat_y = eval_y
