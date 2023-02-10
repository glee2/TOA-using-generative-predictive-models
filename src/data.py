# Notes
'''
Author: Gyumin Lee
Version: 0.6
Description (primary changes): Add classification
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/Transformer/Tech_Gen/'
import sys
sys.path.append(root_dir)

# Basic libraries
import os
import copy
import re
import pandas as pd
import numpy as np

# DL libraries
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import torchvision.datasets
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit, KFold
from sklearn.datasets import load_digits

# Text cleaning libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import cleantext
from cleantext.sklearn import CleanTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOKEN_SOS = "<SOS>"
TOKEN_EOS = "<EOS>"
TOKEN_PAD = "<PAD>"

class TechDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.tokens = [TOKEN_SOS, TOKEN_EOS, TOKEN_PAD]

        for key, value in config.items():
            setattr(self, key, value)

        self.rawdata = pd.read_csv(os.path.join(config.data_dir, "collection_final_with_claims.csv"))
        self.data, self.vocab_w2i, self.vocab_i2w, self.vocab_size, self.seq_len = self.preprocess()
        self.original_idx = np.array(self.data.index)
        self.X, self.Y, self.Y_quantized = self.make_io()

        self.n_classes = len(np.unique(self.Y)) if self.pred_type == "classification" else 1

    def make_io(self, val_main=10, val_sub=1):
        X_df = pd.DataFrame(index=self.data.index)
        if self.data_type == "class":
            X_df['main'] = self.data['main_ipc'].apply(lambda x: self.vocab_w2i[x])
            X_df['sub'] = self.data['sub_ipc'].apply(lambda x: [self.vocab_w2i[xx] for xx in x])
            main_sub_combined = X_df.apply(lambda x: [x['main']]+x['sub'], axis=1)
            X = main_sub_combined.apply(lambda x: np.concatenate([[self.vocab_w2i[TOKEN_SOS]]+x+[self.vocab_w2i[TOKEN_EOS]], np.zeros(self.seq_len-(len(x)+2))+self.vocab_w2i[TOKEN_PAD]]).astype(int))
        elif self.data_type == "claim":
            tokened_claims = self.data['claims'].apply(lambda x: [self.vocab_w2i[xx] for xx in x])
            X = tokened_claims.apply(lambda x: np.concatenate([[self.vocab_w2i[TOKEN_SOS]]+x+[self.vocab_w2i[TOKEN_EOS]], np.zeros(self.seq_len-(len(x)+2))+self.vocab_w2i[TOKEN_PAD]]).astype(int))

        X = np.vstack(X.values)
        Y = self.data['TC'+str(self.n_TC)].values

        Y_quantized = np.zeros_like(Y).astype(int)
        Y_quantized[Y>0] = 1

        if self.pred_type == "classification":
            Y = Y_quantized

        return X, Y, Y_quantized

    def transform(self, sample):
        main_sub_combined = [self.vocab_w2i[sample['main_ipc']]] + [vocab_w2i[i] for i in sample['sub_ipc']]
        X = np.concatenate([main_sub_combined, np.zeros(self.seq_len-(len(main_sub_combined)-2))+self.vocab_w2i[TOKEN_PAD]])
        Y = sample['TC'+str(self.n_TC)]
        return X, Y

    def preprocess(self):
        regex = re.compile("[0-9a-zA-Z\/]+")
        cols_year = ['<1976']+list(np.arange(1976,2018).astype(str))

        if self.data_type == "class":
            rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main ipc','sub ipc'])[['number','main ipc','sub ipc']]
            if self.target_ipc == "ALL":
                main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc'])]
            else:
                main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc']) if self.target_ipc in x]
            rawdata_ipc = rawdata_dropna.loc[rawdata_dropna['main ipc'].isin(main_ipcs)]
            data = rawdata_ipc[['number']].copy(deep=True)
            assert self.ipc_level in [1,2,3], f"Not implemented for an IPC level {self.ipc_level}"
            if self.ipc_level == 1:
                data['main_ipc'] = rawdata_ipc['main ipc'].apply(lambda x: regex.findall(x)[0][:3])
                data['sub_ipc'] = rawdata_ipc['sub ipc'].apply(lambda x: [regex.findall(xx)[0][:3] for xx in x.split(';')])
            elif self.ipc_level == 2:
                data['main_ipc'] = rawdata_ipc['main ipc'].apply(lambda x: regex.findall(x)[0])
                data['sub_ipc'] = rawdata_ipc['sub ipc'].apply(lambda x: [regex.findall(xx)[0] for xx in x.split(';')])
            elif self.ipc_level == 3:
                data['main_ipc'] = rawdata_ipc['main ipc'].apply(lambda x: "".join(regex.findall(x)))
                data['sub_ipc'] = rawdata_ipc['sub ipc'].apply(lambda x: ["".join(regex.findall(xx)) for xx in x.split(';')])
            seq_len = data['sub_ipc'].apply(lambda x: len(x)).max() + 3 # SOS - main ipc - sub ipcs - EOS
        elif self.data_type == "claim":
            rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main ipc','claims'])[['number','main ipc','claims']]
            if self.target_ipc == "ALL":
                main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc'])]
            else:
                main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc']) if self.target_ipc in x]
            assert len(main_ipcs) != 0, "target ipc is not observed"
            rawdata_ipc = rawdata_dropna.loc[rawdata_dropna['main ipc'].isin(main_ipcs)]
            cleaned_claims = self.text_cleaning(text_list=rawdata_ipc['claims'], claim_level=self.claim_level)
            data = pd.concat([rawdata_ipc[['number']], cleaned_claims.rename('claims')], axis=1)
            seq_len = data['claims'].apply(lambda x: len(x)).max() + 3

        rawdata_tc = self.rawdata.loc[rawdata_ipc.index][['year']+cols_year]
        data['TC'+str(self.n_TC)] = rawdata_tc.apply(lambda x: x[np.arange(x['year']+1 if x['year']<2017 else 2017, x['year']+self.n_TC+1 if x['year']+self.n_TC<2018 else 2018).astype(str)].sum(), axis=1)
        data = data.set_index('number')

        if self.data_type == "class":
            main_ipcs = list(np.unique(data['main_ipc']))
            sub_ipcs = list(np.unique(np.concatenate(list(data['sub_ipc'].values))))
            all_items = list(np.union1d(main_ipcs, sub_ipcs))
        elif self.data_type == "claim":
            all_items = list(np.unique(np.concatenate(data['claims'].values)))

        vocab_w2i = {all_items[i]: i for i in range(len(all_items))}
        vocab_w2i.update({self.tokens[i]: len(all_items)+i for i in range(len(self.tokens))})
        vocab_i2w = {i: all_items[i] for i in range(len(all_items))}
        vocab_i2w.update({len(all_items)+i: self.tokens[i] for i in range(len(self.tokens))})
        vocab_size = len(vocab_w2i)

        return (data, vocab_w2i, vocab_i2w, vocab_size, seq_len)

    def text_cleaning(self, text_list=None, claim_level=1, claim_separator="\n\n\n"):
        if not isinstance(text_list, pd.core.series.Series): text_list = pd.Series(text_list)

        basic_cleaner = CleanTransformer(
                        lower=True, no_line_breaks=True, normalize_whitespace=True,
                        no_punct=True, strip_lines=True,
                        no_currency_symbols=True, replace_with_currency_symbol="",
                        no_numbers=True, replace_with_number="",
                        no_digits=True, replace_with_digit="")
        stop_words = stopwords.words("english")
        stemmer = PorterStemmer()

        # Take the first claim
        if claim_level == -1:
            cleaned = text_list
        else:
            cleaned = text_list.apply(lambda x: "".join(x.split(claim_separator)[:claim_level]) if len(x.split(claim_separator))>=claim_level else x)
        # Basic text cleaning
        cleaned = basic_cleaner.transform(cleaned)
        # Remove stopwords
        cleaned = cleaned.apply(lambda claim: np.array([word for word in claim.split() if word not in stop_words]))
        # Stemming
        cleaned = cleaned.apply(lambda claim: [stemmer.stem(word) for word in claim])
        # Remove duplicates and sorting
        cleaned = cleaned.apply(lambda claim: list(np.array(claim)[np.sort(np.unique(claim, return_index=True)[1])]))
        # Remove too frequent or too rare words
        vocab, vocab_counts = np.unique(np.concatenate(cleaned.values), return_counts=True)

        # Set criterion for rare words
        vocab_counts_unique, vocab_counts_frequency = np.unique(vocab_counts, return_counts=True)
        rare_criterion = 5
        sum_frequency = 0
        for i in range(len(vocab_counts_frequency)):
            sum_frequency += vocab_counts_frequency[i]
            if sum_frequency > len(vocab) - 1000:
                rare_criterion = vocab_counts_unique[i]
                break

        freq_words = vocab[np.where(vocab_counts>int(len(cleaned)*0.5))[0]] # frequent words: words that appear more than 40% of the data samples
        rare_words = vocab[np.where(vocab_counts<rare_criterion)[0]] # rare words: words that appear less than 0.5% of the data samples
        freq_rare_set = set(np.concatenate([freq_words, rare_words]))
        print(f"FREQ: {freq_words} ({len(freq_words)}), RARE: {rare_words} ({len(rare_words)})")
        cleaned = cleaned.apply(lambda x: [word for word in x if word not in freq_rare_set])

        return cleaned

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.do_transform:
            X, Y = self.transform(self.data.iloc[idx])
        else:
            X, Y = self.X[idx], self.Y[idx]
        return X, Y

class CVSampler:
    def __init__(self, dataset, test_ratio=0.2, val_ratio=0.2, n_folds=5, random_state=10, stratify=False, oversampled=False):
        self.stratify = stratify
        # self.oversampled = oversampled
        # self.labels = dataset.Y
        self.labels = dataset.Y_quantized
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
