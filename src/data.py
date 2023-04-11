# Notes
'''
Author: Gyumin Lee
Version: 1.0
Description (primary changes): Last version before integration into master branch
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/master/Tech_Gen/'
import sys
sys.path.append(root_dir)

# Basic libraries
import os
import copy
import re
import time
import datetime
import pandas as pd
import numpy as np
from functools import partial

# DL libraries
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import torchvision.datasets
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit, KFold
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
from transformers import DistilBertTokenizer

# Text cleaning libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import cleantext
from cleantext.sklearn import CleanTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TechDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        for key, value in config.items():
            setattr(self, key, value)

        self.rawdata = pd.read_csv(os.path.join(config.data_dir, config.data_file))
        self.target_classes = None
        self.data = self.preprocess()
        self.tokenizer = self.get_tokenizer()
        self.original_idx = np.array(self.data.index)
        self.X, self.Y, self.Y_digitized = self.make_io()
        self.n_outputs = len(np.unique(self.Y)) if self.pred_type == "classification" else 1

    def preprocess(self):
        regex = re.compile("[0-9a-zA-Z\/]+")
        latest_year = datetime.datetime.now().year-1
        cols_year = ['<1976']+list(np.arange(1976,latest_year).astype(str))

        rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main ipc','claims'])[['number','main ipc','sub ipc','claims']]
        if self.target_ipc == "ALL":
            main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc'])]
        else:
            # main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc']) if self.target_ipc in x]
            ## temporarily
            main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc'])]
        assert len(main_ipcs) != 0, "target ipc is not observed"
        data = rawdata_dropna.loc[rawdata_dropna['main ipc'].isin(main_ipcs)]

        rawdata_tc = self.rawdata.loc[data.index]

        ## Get number of forward citations within 5 years (TC5)
        data["TC"+str(self.n_TC)] = rawdata_tc.apply(lambda x: x[np.arange(x["year"] if x["year"]<latest_year else latest_year, x["year"]+1+self.n_TC if x["year"]+1+self.n_TC<latest_year+1 else latest_year).astype(str)].sum(), axis=1)
        data = data.reset_index(drop=True)

        ## Get digitized number of forward citations within 5 years (TC5_digitized)
        bins_criterion = [data["TC"+str(self.n_TC)].quantile(0.9)]
        data["TC"+str(self.n_TC)+"_digitized"] = pd.Series(np.digitize(data["TC"+str(self.n_TC)].values, bins=bins_criterion, right=True))

        ## Get patent class
        patent_classes = data["main ipc"].apply(lambda x: x[:self.class_level]).values
        label_encoder = LabelEncoder()
        label_encoder.fit(patent_classes)
        data["class"] = pd.Series(label_encoder.fit_transform(patent_classes))
        # data["class"] = data["TC"+str(self.n_TC)+"_digitized"]

        if self.target_type == "class":
            self.target_classes = label_encoder.classes_
        elif self.target_type == "citation":
            self.target_classes = ["Least_valuable", "Most_valuable"]

        data = data.set_index('number')

        return data

    def get_tokenizer(self):
        if self.is_pretrained:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            def custom_token_to_id(self): return self.convert_tokens_to_ids
            tokenizer.token_to_id = custom_token_to_id(tokenizer)
            # def custom_get_vocab_size(self): return self.vocab_size
            # tokenizer.get_vocab_size = partial(custom_get_vocab_size, tokenizer)
            def custom_decode_batch(self): return self.batch_decode
            tokenizer.decode_batch = custom_decode_batch(tokenizer)
            def custom_encode_batch(self): return self.encode
            tokenizer.encode_batch = custom_encode_batch(tokenizer)
        else:
            train_tokenizer = False
            tokenizer_path = os.path.join(self.data_dir, f"tokenizer_vocab[{self.vocab_size}].json")
            if self.use_pretrained_tokenizer:
                if os.path.exists(tokenizer_path):
                    tokenizer = Tokenizer.from_file(tokenizer_path)
                else:
                    print("Pretrained tokenizer file is not found, train new tokenizer")
                    train_tokenizer = True
            else:
                train_tokenizer = True

            if train_tokenizer:
                tokenizer = Tokenizer(WordPiece(unk_token="<UNK>"))
                trainer = WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=["<SOS>", "<PAD>", "<EOS>", "<UNK>"], show_progress=True)
                tokenizer.pre_tokenizer = Whitespace()
                tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
                tokenizer.train_from_iterator(self.data['claims'], trainer=trainer)
                tokenizer.post_processor = TemplateProcessing(
                    single="<SOS> $A <EOS>",
                    special_tokens=[
                        ("<SOS>", tokenizer.token_to_id("<SOS>")),
                        ("<EOS>", tokenizer.token_to_id("<EOS>")),
                    ],
                )
                tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>", length=self.max_seq_len)
                if self.max_seq_len > 0:
                    tokenizer.enable_truncation(max_length=self.max_seq_len)
                tokenizer.vocab_size = tokenizer.get_vocab_size()
                tokenizer.save(tokenizer_path)
                print("Tokenizer is trained and saved")

            tokenizer.decoder = decoders.WordPiece()

        return tokenizer

    def extract_keywords(self, claims):
        cleaner = CleanTransformer(
                    lower=True, no_line_breaks=True, normalize_whitespace=True,
                    no_punct=True, replace_with_punct="", strip_lines=True,
                    no_currency_symbols=True, replace_with_currency_symbol="",
                    no_numbers=False, replace_with_number="",
                    no_digits=False, replace_with_digit="")
        stop_words = stopwords.words("english")
        cleaned = [re.sub("([-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·(\\n)])", " ", claim) for claim in claims]
        cleaned = cleaner.transform(cleaned)
        cleaned = [re.sub("(?<!\S)\d+(?!\S)", " ", claim).split() for claim in cleaned] # Also, remove standalone numbers
        cleaned = [" ".join([word for word in claim if word not in stop_words + ["the"]]) for claim in cleaned]

        return cleaned

    def make_io(self, val_main=10, val_sub=1):
        claim_separator = "\n\n\n"
        if self.claim_level != -1:
            X = self.data["claims"].apply(lambda x: "".join(x.split(claim_separator)[:self.claim_level]) if len(x.split(claim_separator))>=self.claim_level else x)
        else:
            X = self.data["claims"]

        if self.target_type == "citation":
            Y_digitized = self.data["TC"+str(self.n_TC)+"_digitized"]
            if self.pred_type == "regression":
                Y = self.data["TC"+str(self.n_TC)]
            elif self.pred_type == "classification":
                Y = self.data["TC"+str(self.n_TC)+"_digitized"]
        elif self.target_type == "class":
            Y = Y_digitized = self.data["class"]
        else:
            Y = None

        return X, Y, Y_digitized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_claims = self.extract_keywords([self.X[idx]])[0] if self.use_keywords else self.X[idx]
        output_claims = self.X[idx]
        if str(self.tokenizer.__class__).split("\'")[1].split(".")[0] == "transformers": # Tokenizer for pretrained language models
            text_inputs = self.tokenizer(input_claims, add_special_tokens=True, max_length=self.max_seq_len, padding="max_length", truncation=True)
            text_inputs_dict = {"input_ids": torch.tensor(text_inputs["input_ids"], dtype=torch.long),
                                "attention_mask": torch.tensor(text_inputs["attention_mask"], dtype=torch.long)}
            if self.use_keywords:
                text_outputs = self.tokenizer(output_claims, add_special_tokens=True, max_length=self.max_seq_len, padding="max_length", truncation=True)
                text_outputs_dict = {"input_ids": torch.tensor(text_outputs["input_ids"], dtype=torch.long),
                                     "attention_mask": torch.tensor(text_outputs["attention_mask"], dtype=torch.long)}
            else:
                text_outputs_dict = text_inputs_dict
        elif str(self.tokenizer.__class__).split("\'")[1].split(".")[0] == "tokenizers": # Custom tokenizer
            text_inputs = self.tokenizer.encode(input_claims) if isinstance(input_claims, str) else self.tokenizer.encode_batch(input_claims)
            text_inputs_dict = {"input_ids": torch.tensor(text_inputs.ids, dtype=torch.long), "attention_mask": torch.tensor(text_inputs.attention_mask, dtype=torch.long)}
            if self.use_keywords:
                text_outputs = self.tokenizer.encode(output_claims)
                text_outputs_dict = {"input_ids": torch.tensor(text_outputs.ids, dtype=torch.long), "attention_mask": torch.tensor(text_outputs.attention_mask, dtype=torch.long)}
            else:
                text_outputs_dict = text_inputs_dict

        target_dtype = torch.long if self.pred_type=="classification" else torch.float32
        target_outputs = torch.tensor(self.Y[idx], dtype=target_dtype)

        out = {"text_inputs": text_inputs_dict, "text_outputs": text_outputs_dict, "targets": target_outputs}

        return out

class CVSampler:
    def __init__(self, dataset, test_ratio=0.2, val_ratio=0.2, n_folds=5, random_state=10, stratify=False, oversampled=False):
        self.stratify = stratify
        self.labels = dataset.Y_digitized
        self.original_idx = dataset.original_idx
        self.dataset = dataset
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.n_folds = n_folds
        self.random_state = random_state

        self.idx_dict = {}
        self.split()

    def split(self):
        if self.stratify:
            test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
            whole_train_idx, test_idx = next(iter(test_splitter.split(np.zeros(len(self.labels)), self.labels)))
            val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.random_state)
            train_idx, val_idx = next(iter(val_splitter.split(np.zeros(len(whole_train_idx)), self.labels[whole_train_idx])))
        else:
            test_splitter = ShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
            whole_train_idx, test_idx = next(iter(test_splitter.split(np.zeros(len(self.labels)), self.labels)))
            val_splitter = ShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.random_state)
            train_idx, val_idx = next(iter(val_splitter.split(np.zeros(len(whole_train_idx)), self.labels[whole_train_idx])))

        self.train_samples_idx = whole_train_idx[train_idx]
        self.val_samples_idx = whole_train_idx[val_idx]
        self.test_samples_idx = test_idx

        if self.n_folds == 1:
            self.idx_dict[0] = {'train': self.train_samples_idx, 'val': self.val_samples_idx, 'test': self.test_samples_idx}
        else:
            if self.stratify:
                kf_splitter = StratifiedKFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
            else:
                kf_splitter = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
            for fold, (train_idx, val_idx) in enumerate(kf_splitter.split(np.zeros(len(whole_train_idx)), self.labels[whole_train_idx])):
                self.idx_dict[fold] = {'train': whole_train_idx[train_idx], 'val': whole_train_idx[val_idx], 'test': self.test_samples_idx}

    def get_idx_dict(self):
        return self.idx_dict

class SMOTESampler:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.n_labels = len(np.unique(self.Y))
        self.n_samples_per_label = np.unique(self.Y, return_counts=True)[1]

    def resampling(self, n_resampled=None, sampling_ratio=None):
        if sampling_ratio is None:
            assert n_resampled, "n_resampled should be specified when sampling_ratio is None"
            sampling_ratio = {}
            for i in range(self.n_labels):
                n_samples = self.n_samples_per_label[i]
                if n_samples > 1:
                    if n_samples < n_resampled:
                        sampling_ratio[i] = n_resampled
                    else:
                        sampling_ratio[i] = n_samples
        print(f"SMOTE Sampling ratio:{sampling_ratio}")

        k_neighbors = np.min(self.n_samples_per_label)-1
        sampler = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k_neighbors)

        X_res, Y_res = sampler.fit_resample(self.X, self.Y)
        oversampled_idx = pd.Index(["SMOTE_oversampled_{}".format(x) for x in range(len(X_res)-len(self.X))])
        X_over = X_res[len(self.X):]
        Y_over = Y_res[len(self.Y):]

        return X_over, Y_over, oversampled_idx
