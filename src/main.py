# Notes
'''
Author: Gyumin Lee
Version: 1.3
Description (primary changes): Claim + class -> class
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/class2class/Tech_Gen/'
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
from models import Transformer, Predictor
from train_utils import EarlyStopping, perf_eval, objective_cv, build_model, train_model, validate_model_mp
from utils import token2class, DotDict, to_device

parser = argparse.ArgumentParser()
## data arguments
parser.add_argument("--data_type", type=str)
parser.add_argument("--target_area", type=str)
parser.add_argument("--ipc_level", type=int)
parser.add_argument("--pred_type", type=str)
parser.add_argument("--use_keywords", default=False, action="store_true")
parser.add_argument("--use_pretrained_tokenizer", default=False, action="store_true")

## training arguments
parser.add_argument("--do_train", default=None, action="store_true")
parser.add_argument("--do_tune", default=None, action="store_true")
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--n_gpus", type=int)
parser.add_argument("--n_folds", type=int)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--mem_verbose", default=False, action="store_true")
parser.add_argument("--alternate_train", default=None, action="store_true")
parser.add_argument("--alternate_train_threshold", default=None, type=int)
parser.add_argument("--weight_alpha", default=None, type=float)

## model arguments
parser.add_argument("--pretrained_enc", default=None, action="store_true")
parser.add_argument("--pretrained_dec", default=None, action="store_true")
parser.add_argument("--use_accelerator", default=None, action="store_true")
parser.add_argument("--n_layers", type=int)
parser.add_argument("--d_embedding", type=int)
parser.add_argument("--d_enc_hidden", type=int)
parser.add_argument("--d_pred_hidden", type=int)
parser.add_argument("--d_ff", type=int)
parser.add_argument("--n_head", type=int)
parser.add_argument("--d_head", type=int)
parser.add_argument("--model_type", type=str)
parser.add_argument("--bidirec", default=None, action="store_true")

## arguments passed only for main.py
parser.add_argument("--do_save", default=False, action="store_true")
parser.add_argument("--do_eval", default=False, action="store_true")
parser.add_argument("--continue_train", default=False, action="store_true")
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--light", default=False, action="store_true")
parser.add_argument("--eval_train_set", default=False, action="store_true")
parser.add_argument("--config_file", default=None, type=str)
parser.add_argument("--analysis_date", default=None, type=str)
parser.add_argument("--use_gpu", default=False, action="store_true")

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

    ''' PART 1: Configuration '''
    project_data_dir = os.path.join(master_dir, "data")
    data_dir = os.path.join("/home2/glee/patent_data/data/")
    model_dir = os.path.join(root_dir, "models")
    result_dir = os.path.join(root_dir, "results")
    config_dir = os.path.join(root_dir, "configs")

    ## parse configuration file
    args = parser.parse_args()
    if args.config_file is not None:
        config_file = args.config_file
    else:
        if args.analysis_date is not None:
            config_file = os.path.join(config_dir, "USED_configs", "[CONFIGS]"+args.analysis_date+".json")
        else:
            config_file = os.path.join(config_dir, "configs_light.json") if args.light else os.path.join(config_dir, "configs.json")
    if args.do_eval: args.do_train = False
    configs = DotDict().load(config_file)
    org_config_keys = {key: list(configs[key].keys()) for key in configs.keys()}

    if args.analysis_date is not None:
        print("Analysis datetime: {}".format(args.analysis_date))
    else:
        print("Analysis datetime: {}".format(current_datetime))
        analysis_date = current_datetime
    
    # parse command line arguments
    instant_configs = {key: value for (key, value) in vars(args).items() if value is not None} # if any argument passed when main.py executed
    instant_configs_for_update = {configkey: {key: value for (key,value) in instant_configs.items() if key in org_config_keys[configkey]} for configkey in org_config_keys.keys()}
    for key, value in configs.items():
        value.update(instant_configs_for_update[key])

    ## assign loss weights
    if configs.model.model_type == "enc-pred-dec":
        configs.train.loss_weights["recon"] = configs.train.loss_weights["recon"] / sum(configs.train.loss_weights.values())
        configs.train.loss_weights["y"] = 1 - configs.train.loss_weights["recon"]
    elif configs.model.model_type == "enc-pred":
        configs.train.loss_weights = {"recon": 0, "y": 1}
    elif configs.model.model_type == "enc-dec":
        configs.train.loss_weights = {"recon": 1, "y": 0}

    ## assign devices
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

    ## extract configurations for dataset
    config_period = "["+"-".join([str(year) for year in configs.data.target_period])+"]"
    config_area = str(configs.data.target_area).replace("\'","").replace(" ","")
    config_keywords = str(configs.data.target_keywords).replace("\'","").replace(" ","")
    config_sampling_ratio = "["+str(configs.data.sampling_ratio)+"sampling"+"]" if configs.data.sampling_ratio < 1 else ""

    ## update configurations
    configs.data.update({"root_dir": root_dir,
                            "data_dir": data_dir,
                            "model_dir": model_dir,
                            "result_dir": result_dir,
                            "pretrained_enc": configs.model.pretrained_enc,
                            "pretrained_dec": configs.model.pretrained_dec,
                            "data_nrows": None,
                            "data_file": "collection_" + "".join([config_keywords, config_area, config_period, config_sampling_ratio]) + ".csv"})
    configs.train.update({"device": device,
                            "device_ids": device_ids,
                            "root_dir": root_dir,
                            "data_dir": data_dir,
                            "model_dir": model_dir,
                            "use_keywords": configs.data.use_keywords,
                            "class_system": configs.data.class_system,
                            "alternate_train_threshold": args.alternate_train_threshold,
                            "curr_ep": 1,
                            "early_stop_patience": int(0.3*configs.train.max_epochs)})
    configs.model.update({"device": device,
                            "device_ids": device_ids,
                            "n_directions": 2 if configs.model.bidirec else 1,
                            "use_accelerator": configs.train.use_accelerator,
                            "model_dir": model_dir})

    ## Set hyperparameters for model training (To be TUNED)
    if configs.train.do_train and configs.train.do_tune:
        n_layers = configs.model.n_layers = None
        d_embedding = configs.model.d_embedding = None
        d_enc_hidden = configs.model.d_enc_hidden = None
        d_pred_hidden = configs.model.d_pred_hidden = None
        learning_rate = configs.train.learning_rate = None
        batch_size = configs.train.batch_size = None
        config_name = "HPARAM_TUNING"
        final_model_path = None
    else:
        n_layers = configs.model.n_layers
        d_embedding = configs.model.d_embedding
        d_enc_hidden = configs.model.d_enc_hidden
        d_pred_hidden = configs.model.d_pred_hidden
        d_latent = configs.model.d_latent

        ## set filename for model
        key_components = {"data": ["class_level", "class_system", "max_seq_len_class", "max_seq_len_claim", "vocab_size"], "model": ["n_layers", "d_hidden", "d_pred_hidden", "d_latent", "d_embedding", "d_ff", "n_head", "d_head"], "train": ["learning_rate", "batch_size", "max_epochs", "curr_ep"]}
        model_config_name_prefix = "".join([config_keywords, config_area, config_period, config_sampling_ratio]) + "data"
        model_config_name = "" + model_config_name_prefix
        model_config_name += f"[{configs.data.class_system}]system"
        for key in ["model", "train"]:
            for component in key_components[key]:
                model_config_name += f"[{str(configs[key][component])}]{component}"
        final_model_path = os.path.join(model_dir, f"[MODEL]{model_config_name}.ckpt")

#     configs.train.update({"model_config_name": model_config_name, "final_model_path": final_model_path})

    ''' PART 2: Dataset setting '''
    tstart = time.time()
    dataset_config_name = "".join([config_keywords, config_area, config_period]) + "data"
    for component in key_components["data"]:
        dataset_config_name += f"[{str(configs.data[component])}]{component}"
    dataset_path = os.path.join(project_data_dir, "pickled_dataset", "[DATASET]"+dataset_config_name+".pickle")

    if os.path.exists(dataset_path) and args.do_save is False:
        print("Load pickled dataset...")
        with open(dataset_path, "rb") as f:
            tech_dataset = pickle.load(f)   # Load pickled dataset if dataset with same configuration already saved
            if tech_dataset.pretrained_enc != configs.data.pretrained_enc or tech_dataset.pretrained_dec != configs.data.pretrained_dec:
                tech_dataset.pretrained_enc = configs.data.pretrained_enc
                tech_dataset.pretrained_dec = configs.data.pretrained_dec
                tech_dataset.tokenizers = tech_dataset.get_tokenizers()
            for tk in tech_dataset.tokenizers.values():
                if "vocab_size" not in dir(tk):
                    tk.vocab_size = tk.get_vocab_size()
            tech_dataset.use_keywords = configs.data.use_keywords
            ## load saved rawdata
            if tech_dataset.rawdata is None:
                tech_dataset.rawdata = pd.read_csv(os.path.join(data_dir, configs.data.data_file), low_memory=False)
        print("Pickled dataset loaded")
    else:
        print("Make dataset...")
        if args.debug:
            configs.data.update({"data_nrows": 5000})
            dataset_path += ".debug"
            configs.train.update({"max_epochs": 5, "batch_size": 100})
        tech_dataset = TechDataset(configs.data)
        if not args.debug:
            rawdata_for_save = copy.deepcopy(tech_dataset.rawdata)
            with open(dataset_path, "wb") as f:
                tech_dataset.rawdata = None
                pickle.dump(tech_dataset, f)
            tech_dataset.rawdata = rawdata_for_save
    tend = time.time()
    print(f"{np.round(tend-tstart,4)} sec elapsed for loading patents for class [{configs.data.target_area}]")

    configs.model.update({"tokenizers": tech_dataset.tokenizers,
                        "n_enc_seq_claim": tech_dataset.max_seq_len_claim,
                        "n_dec_seq_claim": tech_dataset.max_seq_len_claim,
                        "n_enc_seq_class": tech_dataset.max_seq_len_class,
                        "n_dec_seq_class": tech_dataset.max_seq_len_class,
                        "n_outputs": 1 if configs.data.pred_type=="regression" else tech_dataset.n_outputs,
                        "i_padding": tech_dataset.tokenizers["class_enc"].pad_id})

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

            key_components = {"model": ["n_layers", "d_enc_hidden", "d_pred_hidden", "d_latent", "d_embedding", "d_ff", "n_head", "d_head", "take_last_h"], "train": ["learning_rate", "batch_size", "max_epochs", "curr_ep"]}
            config_name_prefix = "".join([config_keywords, config_area, config_period, config_sampling_ratio]) + "data"
            config_name = "" + model_config_name_prefix
            config_name += f"[{configs.data.class_system}]system"
            for key in key_components.keys():
                for component in key_components[key]:
                    config_name += "["+str(configs[key][component])+component+"]"
            final_model_path = os.path.join(model_dir, f"[Final_model][{configs.data.target_area}]{config_name}.ckpt")

            key_components_best = {"data": ["data_type", "pred_type", "target_area", "class_system", "class_level", "claim_level", "n_TC"]}
            config_name_best = ""
            for key in key_components_best.keys():
                for component in key_components_best[key]:
                    config_name_best += "["+str(configs[key][component])+component+"]"
            configs_to_save = {configkey: {key: configs[configkey][key] for key in org_config_keys[configkey]} for configkey in org_config_keys}
            best_config_path = f"[BEST_trial]{config_name_best}.json"
            with open(os.path.join(config_dir, "best_hparam", best_config_path), "w") as f:
                json.dump(configs_to_save, f, indent=4)

        configs_to_save = {configkey: {key: configs[configkey][key] for key in org_config_keys[configkey]} for configkey in org_config_keys}
        fname_configs_to_save = "[CONFIGS]"+current_datetime+".json"

        with open(os.path.join(config_dir, "USED_configs", fname_configs_to_save), "w") as f:
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

        train_loader = DataLoader(train_dataset, batch_size=configs.train.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=configs.train.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

        batch_size_for_test = 16 if len(test_dataset) > 16 else len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_for_test, shuffle=False, num_workers=0)
        batch_size_for_validation = 16 if len(whole_dataset) > 16 else len(whole_dataset)
        whole_loader = DataLoader(whole_dataset, batch_size=batch_size_for_validation, shuffle=False, num_workers=0)

        ## Load best model or train model
        final_model = build_model(configs.model, tokenizers=tech_dataset.tokenizers)
        if re.search("^1.", torch.__version__) is not None:
#             model_size = sum(t.numel() for t in final_model.module.parameters())
            model_size = sum(t.numel() for t in final_model.parameters())
            print(f"Model size: {model_size/1000**2:.1f}M paramaters")
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
            weight_alpha = configs.train.weight_alpha
            class_weights = torch.tensor(np.unique(tech_dataset.Y[whole_idx], return_counts=True)[1][::-1].copy()).to(device)
            class_weights[class_weights.argmax()] = class_weights[class_weights.argmax()] * weight_alpha
            class_weights = class_weights / class_weights.sum()
            if args.continue_train:
                final_model_finder = final_model_path.split("[MODEL]")[-1].split("max_epochs")[0]+"max_epochs"
                matched_ckpts = [f for f in os.listdir(model_dir) if final_model_finder in f]
                latest_ckpt_index = np.argmax([int(f.split("curr_ep")[0].split("[")[-1].replace("]","")) for f in matched_ckpts])
                final_model_path = os.path.join(model_dir, matched_ckpts[latest_ckpt_index])
                if os.path.exists(final_model_path):
                    best_states = torch.load(final_model_path, map_location=device)
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
                curr_ep = int(final_model_path.split("curr_ep")[0].split("[")[-1].replace("]",""))
                configs.train.update({"alternate_train": False, "curr_ep": curr_ep, "max_epochs": configs.train.max_epochs+curr_ep})
            else:
                curr_ep = 1
            (final_model, configs.model, configs.train) = train_model(final_model, train_loader, val_loader, configs.model, configs.train, class_weights=class_weights, model_config_name_prefix=model_config_name_prefix)
        
        model_config_name_prefix = "".join([config_keywords, config_area, config_period, config_sampling_ratio]) + "data"
        model_config_name = "" + model_config_name_prefix
        model_config_name += f"[{configs.data.class_system}]system"
        for key in ["model", "train"]:
            for component in key_components[key]:
                model_config_name += f"[{str(configs[key][component])}]{component}"
        final_model_path = os.path.join(model_dir, f"[MODEL]{model_config_name}.ckpt")

        torch.save(final_model.state_dict(), final_model_path) # Finalize
        torch.cuda.empty_cache()

        print("Training is done!\n")

        fname_datasets_to_save = "[DATASET]"+current_datetime+".xlsx"
        with pd.ExcelWriter(os.path.join(configs.data.result_dir, fname_datasets_to_save)) as writer:
            tech_dataset.data.iloc[whole_idx].to_excel(writer, sheet_name="TRAIN_dataset")
            tech_dataset.data.iloc[test_idx].to_excel(writer, sheet_name="TEST_dataset")
            
        print("Used datasets are saved!\n")

        ''' PART 3-3: Training evaluation '''
        if args.eval_train_set:
            ## Evaluation on train dataset
            print("Validate model on train dataset")
            val_res_train = validate_model_mp(final_model, whole_dataset, mp=mp, model_params=configs.model, train_params=configs.train)
            if "pred" in configs.model.model_type:
                trues_y_train = np.concatenate([res["y"]["true"] for res in val_res_train.values()])
                preds_y_train = np.concatenate([res["y"]["pred"] for res in val_res_train.values()])
                eval_y_train = perf_eval("TRAIN_SET", trues_y_train, preds_y_train, configs=configs, pred_type=configs.data.pred_type, custom_weight=None)
                if configs.data.pred_type == "classification":
                    eval_y_train, confmat_y_train = eval_y_train

            if "dec" in configs.model.model_type:
                trues_recon_train = np.concatenate([res["recon"]["true"] for res in val_res_train.values()])
                preds_recon_train = np.concatenate([res["recon"]["pred"] for res in val_res_train.values()])
                if configs.data.use_keywords:
                    trues_recon_kw_train = np.concatenate([res["recon"]["kw"] for res in val_res_train.values()])
                else:
                    trues_recon_kw_train = np.concatenate([res["recon"]["kw"] for res in val_res_train.values()])
                eval_recon_train = perf_eval("TRAIN_SET", trues_recon_train, preds_recon_train, recon_kw=trues_recon_kw_train, configs=configs, pred_type='generative', tokenizer=final_model.module.tokenizers["class_dec"])
                eval_recon_train.index = pd.Index(list(tech_dataset.data.iloc[whole_idx].index)+[""])
        else:
            eval_recon_train = eval_y_train = confmat_y_train = None

        ## Evaluation on test dataset
        print("Validate model on test dataset")
#         val_res_test = validate_model_mp(final_model, test_dataset, mp=mp, batch_size=4, model_params=configs.model, train_params=configs.train)
        val_res_test = validate_model_mp(final_model_path, test_dataset, mp=mp, batch_size=4, model_params=configs.model, train_params=configs.train, tokenizers=tech_dataset.tokenizers, use_gpu=args.use_gpu, n_cores=configs.train.n_cores)
        if "pred" in configs.model.model_type:
            trues_y_test = np.concatenate([res["y"]["true"] for res in val_res_test.values()])
            preds_y_test = np.concatenate([res["y"]["pred"] for res in val_res_test.values()])
            eval_y_test = perf_eval("TEST_SET", trues_y_test, preds_y_test, configs=configs, pred_type=configs.data.pred_type, custom_weight=None)
            if configs.data.pred_type == "classification":
                eval_y_test, confmat_y_test = eval_y_test
            eval_y_res = pd.concat([eval_y_train, eval_y_test], axis=0)
            if configs.data.pred_type == "classification":
                confmat_y_res = pd.concat([confmat_y_train, confmat_y_test], axis=0)

        if "dec" in configs.model.model_type:
            trues_recon_test = np.concatenate([res["recon"]["true"] for res in val_res_test.values()])
            if configs.data.use_keywords:
                trues_recon_kw_test = np.concatenate([res["recon"]["kw"] for res in val_res_test.values()])
            else:
                trues_recon_kw_test = np.concatenate([res["recon"]["kw"] for res in val_res_test.values()])
            preds_recon_test = np.concatenate([res["recon"]["pred"] for res in val_res_test.values()])
            eval_recon_test = perf_eval("TEST_SET", trues_recon_test, preds_recon_test, recon_kw=trues_recon_kw_test, configs=configs,  pred_type='generative', tokenizer=final_model.module.tokenizers["class_dec"])
            eval_recon_test.index = pd.Index(list(tech_dataset.data.iloc[test_idx].index)+[""])

        fname_results_to_save = "[RESULTS]"+current_datetime+".xlsx"

        with pd.ExcelWriter(os.path.join(configs.data.result_dir, fname_results_to_save)) as writer:
            if "pred" in configs.model.model_type:
                eval_y_res.to_excel(writer, sheet_name=f"{configs.data.pred_type}_metrics")
                if configs.data.pred_type == "classification":
                    confmat_y_res.to_excel(writer, sheet_name="Confusion_matrix")
            if "dec" in configs.model.model_type:
                if args.eval_train_set:
                    eval_recon_train.to_excel(writer, sheet_name="Generative_TRAIN")
                eval_recon_test.to_excel(writer, sheet_name="Generative_TEST")

        torch.cuda.empty_cache()

        with open(os.path.join(configs.data.result_dir, "USED_configs", fname_configs_to_save), "w") as f:
            json.dump(configs_to_save, f, indent=4)

    else:
        final_model = build_model(configs.model, tokenizers=tech_dataset.tokenizers)
        final_model_finder = final_model_path.split("[MODEL]")[-1].split("max_epochs")[0]+"max_epochs"
        matched_ckpts = [f for f in os.listdir(model_dir) if final_model_finder in f]
        latest_ckpt_index = np.argmax([int(f.split("curr_ep")[0].split("[")[-1].replace("]","")) for f in matched_ckpts])
        final_model_path = os.path.join(model_dir, matched_ckpts[latest_ckpt_index])
        if os.path.exists(final_model_path):
            best_states = torch.load(final_model_path, map_location=device)
        else:
            raise Exception("Model need to be trained first")
            
        has_module_prefix = any(k.startswith("module.") for k in best_states.keys())
        if has_module_prefix:
            stripped = {}
            for k, v in best_states.items():
                new_key = k[len("module."):] if k.startswith("module.") else k
                stripped[new_key] = v
            best_states = stripped
        final_model.load_state_dict(best_states)

        del best_states
        torch.cuda.empty_cache()
        print("Model successfully loaded")

        if args.analysis_date is not None:
            analysis_date = args.analysis_date

            used_train_data = pd.read_excel(os.path.join(result_dir, "[DATASET]"+analysis_date+".xlsx"), sheet_name="TRAIN_dataset")
            train_idx = tech_dataset.data.index.astype(int).get_indexer(pd.Index(used_train_data["patent_number"]))
            train_dataset = Subset(tech_dataset, train_idx)

            used_test_data = pd.read_excel(os.path.join(result_dir, "[DATASET]"+analysis_date+".xlsx"), sheet_name="TEST_dataset")
            test_idx = tech_dataset.data.index.astype(int).get_indexer(pd.Index(used_test_data["patent_number"]))
            test_dataset = Subset(tech_dataset, test_idx)

            configs_to_save = {configkey: {key: configs[configkey][key] for key in org_config_keys[configkey]} for configkey in org_config_keys}
            fname_configs_to_save = "[CONFIGS]"+analysis_date+".json"

            if args.eval_train_set:
                ## Evaluation on train dataset
                print("Validate model on train dataset")
#                 val_res_train = validate_model_mp(final_model, train_dataset, mp=mp, model_params=configs.model, train_params=configs.train)
                val_res_train = validate_model_mp(final_model_path, train_dataset, mp=mp, batch_size=4, model_params=configs.model, train_params=configs.train, tokenizers=tech_dataset.tokenizers, use_gpu=False, n_cores=configs.train.n_cores)
                if "pred" in configs.model.model_type:
                    trues_y_train = np.concatenate([res["y"]["true"] for res in val_res_train.values()])
                    preds_y_train = np.concatenate([res["y"]["pred"] for res in val_res_train.values()])
                    eval_y_train = perf_eval("TRAIN_SET", trues_y_train, preds_y_train, configs=configs, pred_type=configs.data.pred_type, custom_weight=None)
                    if configs.data.pred_type == "classification":
                        eval_y_train, confmat_y_train = eval_y_train

                if "dec" in configs.model.model_type:
                    trues_recon_train = np.concatenate([res["recon"]["true"] for res in val_res_train.values()])
                    preds_recon_train = np.concatenate([res["recon"]["pred"] for res in val_res_train.values()])
                    if configs.data.use_keywords:
                        trues_recon_kw_train = np.concatenate([res["recon"]["kw"] for res in val_res_train.values()])
                    else:
                        trues_recon_kw_train = np.concatenate([res["recon"]["kw"] for res in val_res_train.values()])
                    eval_recon_train = perf_eval("TRAIN_SET", trues_recon_train, preds_recon_train, recon_kw=trues_recon_kw_train, configs=configs, pred_type='generative', tokenizer=final_model.module.tokenizers["class_dec"])
                    eval_recon_train.index = pd.Index(list(tech_dataset.data.iloc[train_idx].index)+[""])
            else:
                eval_recon_train = eval_y_train = confmat_y_train = None

            ## Evaluation on test dataset
            print("Validate model on test dataset")
#             val_res_test = validate_model_mp(final_model, test_dataset, mp=mp, batch_size=4, model_params=configs.model, train_params=configs.train)
            val_res_test = validate_model_mp(final_model_path, test_dataset, mp=mp, batch_size=4, model_params=configs.model, train_params=configs.train, tokenizers=tech_dataset.tokenizers, use_gpu=False, n_cores=configs.train.n_cores)
            if "pred" in configs.model.model_type:
                trues_y_test = np.concatenate([res["y"]["true"] for res in val_res_test.values()])
                preds_y_test = np.concatenate([res["y"]["pred"] for res in val_res_test.values()])
                eval_y_test = perf_eval("TEST_SET", trues_y_test, preds_y_test, configs=configs, pred_type=configs.data.pred_type, custom_weight=None)
                if configs.data.pred_type == "classification":
                    eval_y_test, confmat_y_test = eval_y_test
                eval_y_res = pd.concat([eval_y_train, eval_y_test], axis=0)
                if configs.data.pred_type == "classification":
                    confmat_y_res = pd.concat([confmat_y_train, confmat_y_test], axis=0)

            if "dec" in configs.model.model_type:
                trues_recon_test = np.concatenate([res["recon"]["true"] for res in val_res_test.values()])
                if configs.data.use_keywords:
                    trues_recon_kw_test = np.concatenate([res["recon"]["kw"] for res in val_res_test.values()])
                else:
                    trues_recon_kw_test = np.concatenate([res["recon"]["kw"] for res in val_res_test.values()])
                preds_recon_test = np.concatenate([res["recon"]["pred"] for res in val_res_test.values()])
                eval_recon_test = perf_eval("TEST_SET", trues_recon_test, preds_recon_test, recon_kw=trues_recon_kw_test, configs=configs,  pred_type='generative', tokenizer=tech_dataset.tokenizers["class_dec"])
                eval_recon_test.index = pd.Index(list(tech_dataset.data.iloc[test_idx].index)+[""])

            fname_results_to_save = "[RESULTS]"+current_datetime+".xlsx"
            fname_datasets_to_save = "[DATASET]"+current_datetime+".xlsx"

            with pd.ExcelWriter(os.path.join(configs.data.result_dir, fname_results_to_save)) as writer:
                if "pred" in configs.model.model_type:
                    eval_y_res.to_excel(writer, sheet_name=f"{configs.data.pred_type}_metrics")
                    if configs.data.pred_type == "classification":
                        confmat_y_res.to_excel(writer, sheet_name="Confusion_matrix")
                if "dec" in configs.model.model_type:
                    if args.eval_train_set:
                        eval_recon_train.to_excel(writer, sheet_name="Generative_TRAIN")
                    eval_recon_test.to_excel(writer, sheet_name="Generative_TEST")

            with pd.ExcelWriter(os.path.join(configs.data.result_dir, fname_datasets_to_save)) as writer:
                tech_dataset.data.iloc[train_idx].to_excel(writer, sheet_name="TRAIN_dataset")
                tech_dataset.data.iloc[test_idx].to_excel(writer, sheet_name="TEST_dataset")

            torch.cuda.empty_cache()

            with open(os.path.join(config_dir, "USED_configs", fname_configs_to_save), "w") as f:
                json.dump(configs_to_save, f, indent=4)

            with open(os.path.join(configs.data.result_dir, "USED_configs", fname_configs_to_save), "w") as f:
                json.dump(configs_to_save, f, indent=4)

        else:
            instant_dataset = Subset(tech_dataset, np.random.choice(np.arange(len(tech_dataset)), 1000))
            data_loader = DataLoader(instant_dataset, batch_size=128)

            print("Validate model on test dataset")
            val_res = validate_model_mp(final_model, instant_dataset, mp=mp, batch_size=64, model_params=configs.model, train_params=configs.train)
            trues_recon = np.concatenate([res["recon"]["true"] for res in val_res.values()])
            preds_recon = np.concatenate([res["recon"]["pred"] for res in val_res.values()])
            trues_y = np.concatenate([res["y"]["true"] for res in val_res.values()])
            preds_y = np.concatenate([res["y"]["pred"] for res in val_res.values()])
            
            if configs.data.use_keywords:
                trues_recon_kw = np.concatenate([res["recon"]["kw"] for res in val_res.values()])
            else:
                trues_recon_kw = np.concatenate([res["recon"]["kw"] for res in val_res.values()])

            eval_recon = perf_eval("LOADED_MODEL", trues_recon, preds_recon, recon_kw=trues_recon_kw, configs=configs, pred_type='generative', tokenizer=tech_dataset.tokenizers["class_dec"])
            eval_y = perf_eval("LOADED_MODEL", trues_y, preds_y, configs=configs, pred_type=configs.data.pred_type)
            if configs.data.pred_type == "classification":
                eval_y, confmat_y = eval_y

            fname_results_to_save = "[LOADED_RESULTS]"+analysis_date+".xlsx"

            with pd.ExcelWriter(os.path.join(configs.data.result_dir, fname_results_to_save)) as writer:
                eval_y.to_excel(writer, sheet_name=configs.data.pred_type+" results")
                eval_recon.to_excel(writer, sheet_name="Generation results")

def instant_save(model):
    torch.save(model.state_dict(), final_model_path)
    with open(os.path.join(config_dir, "USED_configs", fname_configs_to_save), "w") as f:
        json.dump(configs_to_save, f, indent=4)
                
def save_successful():
    successful_path = os.path.join('/home2/glee/dissertation/1_tech_gen_impact/class2class/successful_backups/', current_datetime)
    if not os.path.exists(successful_path):
        os.mkdir(successful_path)
    successful_final_model_path = os.path.join(successful_path, f"[Final_model]{model_config_name}.ckpt")
    torch.save(final_model.state_dict(), successful_final_model_path)
    with pd.ExcelWriter(os.path.join(successful_path, fname_results_to_save)) as writer:
        if "pred" in configs.model.model_type:
            eval_y_res.to_excel(writer, sheet_name=f"{configs.data.pred_type}_metrics")
            if configs.data.pred_type == "classification":
                confmat_y_res.to_excel(writer, sheet_name="Confusion_matrix")
        if "dec" in configs.model.model_type:
            if args.eval_train_set:
                eval_recon_train.to_excel(writer, sheet_name="Generative_TRAIN")
            eval_recon_test.to_excel(writer, sheet_name="Generative_TEST")
    with pd.ExcelWriter(os.path.join(successful_path, fname_datasets_to_save)) as writer:
        if args.do_eval and args.analysis_date is not None:
            tech_dataset.data.iloc[train_idx].to_excel(writer, sheet_name="TRAIN_dataset")
        else:
            tech_dataset.data.iloc[whole_idx].to_excel(writer, sheet_name="TRAIN_dataset")
        tech_dataset.data.iloc[test_idx].to_excel(writer, sheet_name="TEST_dataset")
    with open(os.path.join(successful_path, fname_configs_to_save), "w") as f:
        json.dump(configs_to_save, f, indent=4)
