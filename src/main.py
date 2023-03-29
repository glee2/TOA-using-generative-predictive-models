# Notes
'''
Author: Gyumin Lee
Version: 1.0
Description (primary changes): Last version before integration into master branch
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/master/Tech_Gen/'
master_dir = '/home2/glee/dissertation/1_tech_gen_impact/master/Tech_Gen/'
import sys
sys.path.append(root_dir)

import copy
import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import math
import time
import pickle
import re
import json
import datetime
import multiprocess as mp
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
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from data import TechDataset, CVSampler
from models import Transformer
from train_utils import EarlyStopping, perf_eval, objective_cv, build_model, train_model, validate_model_mp
from utils import token2class, DotDict

parser = argparse.ArgumentParser()
## data arguments
parser.add_argument("--data_type", type=str)
parser.add_argument("--target_ipc", type=str)
parser.add_argument("--pred_type", type=str)
parser.add_argument("--use_pretrained_tokenizer", default=False, action="store_true")

## training arguments
parser.add_argument("--do_train", default=None, action="store_true")
parser.add_argument("--do_tune", default=None, action="store_true")
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--n_folds", type=int)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--mem_verbose", default=False, action="store_true")
parser.add_argument("--alternate_train", default=False, action="store_true")

## model arguments
parser.add_argument("--use_accelerator", default=None, action="store_true")
parser.add_argument("--n_layers", type=int)
parser.add_argument("--d_embedding", type=int)
parser.add_argument("--d_hidden", type=int)
parser.add_argument("--d_ff", type=int)
parser.add_argument("--n_head", type=int)
parser.add_argument("--d_head", type=int)
parser.add_argument("--model_type", type=str)
parser.add_argument("--bidirec", default=False, action="store_true")

## arguments passed only for main.py
parser.add_argument("--do_save", default=False, action="store_true")
parser.add_argument("--light", default=False, action="store_true")
parser.add_argument("--eval_train_set", default=False, action="store_true")
parser.add_argument("--config_file", default=None, type=str)

if __name__=="__main__":
    mp.set_start_method("spawn")

    ''' PART 1: Configuration '''
    data_dir = os.path.join(master_dir, "data")
    model_dir = os.path.join(root_dir, "models")
    result_dir = os.path.join(root_dir, "results")
    config_dir = os.path.join(root_dir, "configs")

    args = parser.parse_args()
    if args.config_file is not None:
        config_file = args.config_file
    else:
        config_file = os.path.join(config_dir, "configs_light.json") if args.light else os.path.join(config_dir, "configs.json")
    configs = DotDict().load(config_file)
    org_config_keys = {key: list(configs[key].keys()) for key in configs.keys()}

    instant_configs = {key: value for (key, value) in vars(args).items() if value is not None} # if any argument passed when main.py executed
    instant_configs_for_update = {configkey: {key: value for (key,value) in instant_configs.items() if key in org_config_keys[configkey]} for configkey in org_config_keys.keys()}
    for key, value in configs.items():
        value.update(instant_configs_for_update[key])

    regex_ipc = re.compile('[A-Z](?![\\D])')
    if regex_ipc.match(configs.data.target_ipc) is None:
        configs.data.update({"target_ipc": "ALL"})
    elif len(configs.data.target_ipc) > 5:
        configs.data.update({"target_ipc": configs.data.target_ipc[:4]})

    if configs.model.model_type == "enc-pred-dec":
        configs.train.loss_weights["recon"] = configs.train.loss_weights["recon"] / sum(configs.train.loss_weights.values())
        configs.train.loss_weights["y"] = 1 - configs.train.loss_weights["recon"]
    elif configs.model.model_type == "enc-pred":
        configs.train.loss_weights = {"recon": 0, "y": 1}
    elif configs.model.model_type == "enc-dec":
        configs.train.loss_weights = {"recon": 1, "y": 0}

    if configs.train.use_accelerator:
        accelerator = Accelerator()
        device_ids = list(range(torch.cuda.device_count()))
        device = accelerator.device

        configs.train.update({"accelerator": accelerator})
    else:
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            gpu_usages = [np.sum([float(usage.split("uses")[-1].replace(" ","").replace("MB","")) for usage in torch.cuda.list_gpu_processes(id).split("GPU memory") if not usage=="" and "no processes are running" not in usage]) for id in device_ids]
            device_ids = np.argsort(gpu_usages)[:configs.train.n_gpus]
            device_ids = list(map(lambda x: torch.device('cuda', x),list(device_ids)))
            device = device_ids[0] # main device
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
            device_ids = []

    configs.data.update({"root_dir": root_dir,
                            "data_dir": data_dir,
                            "model_dir": model_dir,
                            "result_dir": result_dir})
    configs.train.update({"device": device,
                            "device_ids": device_ids,
                            "root_dir": root_dir,
                            "data_dir": data_dir,
                            "model_dir": model_dir,
                            "use_keywords": configs.data.use_keywords,
                            "early_stop_patience": int(0.3*configs.train.max_epochs)})
    configs.model.update({"device": device,
                            "device_ids": device_ids,
                            "n_directions": 2 if configs.model.bidirec else 1,
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
        d_latent = configs.model.n_enc_seq * configs.model.d_hidden

        key_components = {"data": ["target_ipc", "pred_type", "max_seq_len", "vocab_size"], "model": ["n_layers", "d_hidden", "d_embedding", "d_ff", "n_head", "d_head"], "train": ["learning_rate", "batch_size", "max_epochs"]}
        config_name = ""
        for key in key_components.keys():
            for component in key_components[key]:
                config_name += "["+str(configs[key][component])+component+"]"
        final_model_path = os.path.join(model_dir, f"[Final_model]{config_name}.ckpt")

    configs.model.update({"d_latent": d_latent})
    configs.train.update({"config_name": config_name,
                            "final_model_path": final_model_path})

    ''' PART 2: Dataset setting '''
    tstart = time.time()
    dataset_config_name = "-".join([str(key)+"="+str(value) for (key,value) in configs.data.items() if key in org_config_keys["data"]])
    dataset_path = os.path.join(data_dir, "pickled_dataset", "[tech_dataset]"+dataset_config_name+".pickle")
    if os.path.exists(dataset_path) and args.do_save is False:
        print("Load pickled dataset...")
        with open(dataset_path, "rb") as f:
            tech_dataset = pickle.load(f)   # Load pickled dataset if dataset with same configuration already saved
        print("Pickled dataset loaded")
    else:
        print("Make dataset...")
        tech_dataset = TechDataset(configs.data)
        with open(dataset_path, "wb") as f:
            tech_dataset.rawdata = None
            pickle.dump(tech_dataset, f)
    tend = time.time()
    print(f"{np.round(tend-tstart,4)} sec elapsed for loading patents for class [{configs.data.target_ipc}]")

    configs.model.update({"tokenizer": tech_dataset.tokenizer,
                        "n_enc_vocab": tech_dataset.tokenizer.get_vocab_size(),
                        "n_dec_vocab": tech_dataset.tokenizer.get_vocab_size(),
                        "n_enc_seq": tech_dataset.max_seq_len,
                        "n_dec_seq": tech_dataset.max_seq_len,
                        "n_outputs": 1 if configs.data.pred_type=="regression" else tech_dataset.n_outputs,
                        "i_padding": tech_dataset.tokenizer.token_to_id("<PAD>")})
    if not configs.train.do_tune:
        configs.model.update({"d_latent": configs.model.n_enc_seq * configs.model.d_hidden})

    ''' PART 3: Training '''
    if configs.train.do_train:
        sampler = CVSampler(tech_dataset, n_folds=configs.train.n_folds, test_ratio=0.1, stratify=True)
        cv_idx = sampler.get_idx_dict()
        print(f"#Samples\nTrain: {len(cv_idx[0]['train'])}, Validation: {len(cv_idx[0]['val'])}, Test: {len(cv_idx[0]['test'])}")

        ''' PART 3-1: Hyperparmeter tuning '''
        if configs.train.do_tune:
            configs.train.update({"tuned_model_path": os.path.join(model_dir,"hparam_tuning")})
            optuna_obj = lambda trial: objective_cv(trial, dataset=tech_dataset, cv_idx=cv_idx, model_params=configs.model, train_params=configs.train)

            opt_sampler = TPESampler()
            study = optuna.create_study(direction='minimize')
            study.optimize(optuna_obj, n_trials=configs.train.n_trials, gc_after_trial=True, show_progress_bar=True, callbacks=[lambda study, trial: torch.cuda.empty_cache()])
            best_params = study.best_trial.params

            print(f"Best trial:\n  CrossEntropyLoss: {study.best_trial.value}\n  Params:")
            for k, v in best_params.items():
                print(f"    {k}: {v}")
                if isinstance(v, np.int64): best_params[k] = int(v)

            configs.train.update({k: v for k,v in best_params.items() if k in configs.train.keys()})
            configs.model.update({k: v for k,v in best_params.items() if k in configs.model.keys()})

            key_components = {"model": ["n_layers", "d_hidden", "d_embedding", "d_ff", "n_head", "d_head"], "train": ["learning_rate", "batch_size", "max_epochs"]}
            config_name = ""
            for key in key_components.keys():
                for component in key_components[key]:
                    config_name += "["+str(configs[key][component])+component+"]"
            final_model_path = os.path.join(model_dir, f"[Final_model][{configs.data.target_ipc}]{config_name}.ckpt")

            key_components_best = {"data": ["data_type", "pred_type", "target_ipc", "ipc_level", "claim_level", "n_TC"]}
            config_name_best = ""
            for key in key_components_best.keys():
                for component in key_components_best[key]:
                    config_name_best += "["+str(configs[key][component])+component+"]"
            configs_to_save = {configkey: {key: configs[configkey][key] for key in org_config_keys[configkey]} for configkey in org_config_keys}
            best_config_path = f"[BEST_trial]{config_name_best}.json"
            with open(os.path.join(config_dir, "best_hparam", best_config_path), "w") as f:
                json.dump(configs_to_save, f, indent=4)

        ''' PART 3-2: Dataset construction and model training '''
        ## Construct datasets
        train_idx = cv_idx[0]['train']
        val_idx = cv_idx[0]['val']
        test_idx = cv_idx[0]['test']
        whole_idx = np.concatenate([train_idx, val_idx])

        train_dataset = Subset(tech_dataset, train_idx)
        val_dataset = Subset(tech_dataset, val_idx)
        test_dataset = Subset(tech_dataset, test_idx)
        whole_dataset = Subset(tech_dataset, whole_idx)

        train_loader = DataLoader(train_dataset, batch_size=configs.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=configs.train.batch_size, shuffle=True, num_workers=0, drop_last=True)

        batch_size_for_test = 128 if len(test_dataset) > 128 else len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_for_test, shuffle=False, num_workers=0)
        batch_size_for_validation = 128 if len(whole_dataset) > 128 else len(whole_dataset)
        whole_loader = DataLoader(whole_dataset, batch_size=batch_size_for_validation, shuffle=False, num_workers=0)

        ## Load best model or train model
        final_model = build_model(configs.model, tokenizer=tech_dataset.tokenizer)
        if re.search("^1.", torch.__version__) is not None:
            with torch.no_grad():
                if configs.data.use_keywords:
                    x_input, x_keywords_input, _ = tech_dataset.__getitem__(0)
                else:
                    x_input, _ = tech_dataset.__getitem__(0)
                x_input = torch.tensor(x_input, device=device).unsqueeze(0)
                print(pytorch_model_summary.summary(final_model.module, torch.zeros(x_input.shape, device=device, dtype=torch.long), torch.zeros(x_input.shape, device=device, dtype=torch.long), show_input=True, max_depth=None, show_parent_layers=True))
        else:
            print("INFO: pytorch-model-summary does not support PyTorch 2.0, so just print model structure")
            print(final_model)
            torch._dynamo.config.verbose = True
            torch._dynamo.config.suppress_errors = True

        if configs.train.do_tune:
            best_states = torch.load(os.path.join(configs.train.model_dir,f"[HPARAM_TUNING]{study.best_trial.number}trial.ckpt"))
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

        torch.cuda.empty_cache()

        print("Training is done!\n")

        ''' PART 3-3: Training evaluation '''
        if args.eval_train_set:
            ## Evaluation on train dataset
            print("Validate model on train dataset")
            val_res_train = validate_model_mp(final_model, whole_dataset, mp=mp, model_params=configs.model, train_params=configs.train)
            trues_recon_train = np.concatenate([res["recon"]["true"] for res in val_res_train.values()])
            preds_recon_train = np.concatenate([res["recon"]["pred"] for res in val_res_train.values()])
            trues_y_train = np.concatenate([res["y"]["true"] for res in val_res_train.values()])
            preds_y_train = np.concatenate([res["y"]["pred"] for res in val_res_train.values()])

            eval_recon_train = perf_eval("TRAIN_SET", trues_recon_train, preds_recon_train, configs=configs, pred_type='generative', tokenizer=final_model.module.tokenizer)
            eval_recon_train.index = pd.Index(list(tech_dataset.data.iloc[whole_idx].index)+[""])
            # eval_recon_train = perf_eval("TRAIN_SET", trues_recon_train, preds_recon_train, configs=configs, pred_type='generative', tokenizer=final_model.module.tokenizer)
            eval_y_train = perf_eval("TRAIN_SET", trues_y_train, preds_y_train, configs=configs, pred_type=configs.data.pred_type)
            if configs.data.pred_type == "classification":
                eval_y_train, confmat_y_train = eval_y_train
        else:
            eval_recon_train = eval_y_train = confmat_y_train = None

        ## Evaluation on test dataset
        print("Validate model on test dataset")
        val_res_test = validate_model_mp(final_model, test_dataset, mp=mp, batch_size=64, model_params=configs.model, train_params=configs.train)
        if "pred" in configs.model.model_type:
            trues_y_test = np.concatenate([res["y"]["true"] for res in val_res_test.values()])
            preds_y_test = np.concatenate([res["y"]["pred"] for res in val_res_test.values()])
            eval_y_test = perf_eval("TEST_SET", trues_y_test, preds_y_test, configs=configs, pred_type=configs.data.pred_type)
            if configs.data.pred_type == "classification":
                eval_y_test, confmat_y_test = eval_y_test
            eval_y_res = pd.concat([eval_y_train, eval_y_test], axis=0)
            if configs.data.pred_type == "classification":
                confmat_y_res = pd.concat([confmat_y_train, confmat_y_test], axis=0)

        if "dec" in configs.model.model_type:
            trues_recon_test = np.concatenate([res["recon"]["true"] for res in val_res_test.values()])
            preds_recon_test = np.concatenate([res["recon"]["pred"] for res in val_res_test.values()])
            eval_recon_test = perf_eval("TEST_SET", trues_recon_test, preds_recon_test, configs=configs,  pred_type='generative', tokenizer=final_model.module.tokenizer)
            eval_recon_test.index = pd.Index(list(tech_dataset.data.iloc[test_idx].index)+[""])
            eval_recon_res = pd.concat([eval_recon_train, eval_recon_test], axis=0)

        with pd.ExcelWriter(os.path.join(configs.data.result_dir,f"[TRAIN-RESULT][{configs.data.target_ipc}]{configs.train.config_name}.xlsx")) as writer:
            if "pred" in configs.model.model_type:
                eval_y_res.to_excel(writer, sheet_name=f"{configs.data.pred_type}_metrics")
                if configs.data.pred_type == "classification":
                    confmat_y_res.to_excel(writer, sheet_name="Confusion_matrix")
            if "dec" in configs.model.model_type:
                if args.eval_train_set:
                    eval_recon_train.to_excel(writer, sheet_name="Generative_TRAIN")
                    tech_dataset.data.iloc[whole_idx].to_excel(writer, sheet_name="TRAIN_dataset")
                eval_recon_test.to_excel(writer, sheet_name="Generative_TEST")
                tech_dataset.data.iloc[test_idx].to_excel(writer, sheet_name="TEST_dataset")

        torch.cuda.empty_cache()

        configs_to_save = {configkey: {key: configs[configkey][key] for key in org_config_keys[configkey]} for configkey in org_config_keys}
        fname_config_to_save = "[CONFIGS]"+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")+".json"
        with open(os.path.join(config_dir, fname_config_to_save), "w") as f:
            json.dump(configs_to_save, f, indent=4)

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

        del best_states
        del converted_states
        torch.cuda.empty_cache()

        instant_dataset = Subset(tech_dataset, np.random.choice(np.arange(len(tech_dataset)), 1000))
        data_loader = DataLoader(instant_dataset, batch_size=128)

        val_res = validate_model_mp(final_model, instant_dataset, mp=mp, batch_size=64, model_params=configs.model, train_params=configs.train)
        trues_recon = np.concatenate([res["recon"]["true"] for res in val_res.values()])
        preds_recon = np.concatenate([res["recon"]["pred"] for res in val_res.values()])
        trues_y = np.concatenate([res["y"]["true"] for res in val_res.values()])
        preds_y = np.concatenate([res["y"]["pred"] for res in val_res.values()])

        eval_recon = perf_eval("LOADED_MODEL", trues_recon, preds_recon, configs=configs, pred_type='generative', tokenizer=configs.model.tokenizer)
        eval_y = perf_eval("LOADED_MODEL", trues_y, preds_y, configs=configs, pred_type=configs.data.pred_type)
        if configs.data.pred_type == "classification":
            eval_y, confmat_y = eval_y

        with pd.ExcelWriter(os.path.join(configs.data.result_dir,f"[LOADED-RESULT]{configs.train.config_name}.xlsx")) as writer:
            eval_y.to_excel(writer, sheet_name=configs.data.pred_type+" results")
            eval_recon.to_excel(writer, sheet_name="Generation results")
