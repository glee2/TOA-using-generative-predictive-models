# Notes
'''
Author: Gyumin Lee
Version: 0.4
Description (primary changes): Hyperparameter tuning
'''

# Set root directory
root_dir = '/home2/glee/Tech_Gen/'
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
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit, KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_PAD = '<PAD>'

regex = re.compile("[0-9a-zA-Z\/]+")

class TechDataset(Dataset):
    def __init__(self, device=None, data_dir="", params=None, do_transform=False):
        super().__init__()
        self.tokens = [TOKEN_SOS, TOKEN_EOS, TOKEN_PAD]
        self.device = device
        self.data_dir = data_dir
        if params is not None:
            self.params = {'target_ipc': 'A61C', 'n_TC': 3, 'ipc_level': 3}
            self.params.update(params)
        else:
            self.params = {'target_ipc': 'A61C', 'n_TC': 3, 'ipc_level': 3}
        self.do_transform = do_transform

        self.rawdata = pd.read_csv(os.path.join(data_dir, "collection_final.csv"))

        self.data, self.vocab_w2i, self.vocab_i2w, self.vocab_size, self.seq_len = self.preprocess(target_ipc=self.params['target_ipc'], ipc_level=self.params['ipc_level'], n_TC=self.params['n_TC'])
        self.original_idx = np.array(self.data.index)
        self.X, self.Y = self.make_io()

        self.oversampled_idx = self.resampled_idx = np.array([])

    def make_io(self, val_main=10, val_sub=1):
        X_df = pd.DataFrame(index=self.data.index)
        X_df['main'] = self.data['main_ipc'].apply(lambda x: self.vocab_w2i[x])
        X_df['sub'] = self.data['sub_ipc'].apply(lambda x: [self.vocab_w2i[xx] for xx in x])
        main_sub_combined = X_df.apply(lambda x: [x['main']]+x['sub'], axis=1)
        X_df['seq'] = main_sub_combined.apply(lambda x: np.concatenate([[self.vocab_w2i[TOKEN_SOS]]+x+[self.vocab_w2i[TOKEN_EOS]], np.zeros(self.seq_len-(len(x)+2))+self.vocab_w2i[TOKEN_PAD]]).astype(int))

        X = np.vstack(X_df['seq'].values)
        Y = self.data['TC'+str(self.params['n_TC'])].values

        return X, Y

    def transform(self, sample):
        main_sub_combined = [self.vocab_w2i[sample['main_ipc']]] + [vocab_w2i[i] for i in sample['sub_ipc']]
        X = np.concatenate([main_sub_combined, np.zeros(self.seq_len-(len(main_sub_combined)-2))+self.vocab_w2i[TOKEN_PAD]])
        Y = sample['TC'+str(self.params['n_TC'])]
        return X, Y

    def preprocess(self, target_ipc='A61C', ipc_level=3, n_TC=3):
        cols_year = ['<1976']+list(np.arange(1976,2018).astype(str))

        rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main ipc', 'sub ipc'])[['number','main ipc','sub ipc']]
        main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc']) if target_ipc in x]
        rawdata_ipc = rawdata_dropna.loc[rawdata_dropna['main ipc'].isin(main_ipcs)]
        rawdata_tc = self.rawdata.loc[rawdata_ipc.index][['year']+cols_year]

        data = rawdata_ipc[['number']].copy(deep=True)
        assert ipc_level in [1,2,3], f"Not implemented for an IPC level {ipc_level}"
        if ipc_level == 1:
            data['main_ipc'] = rawdata_ipc['main ipc'].apply(lambda x: regex.findall(x)[0][:3])
            data['sub_ipc'] = rawdata_ipc['sub ipc'].apply(lambda x: [regex.findall(xx)[0][:3] for xx in x.split(';')])
        elif ipc_level == 2:
            data['main_ipc'] = rawdata_ipc['main ipc'].apply(lambda x: regex.findall(x)[0])
            data['sub_ipc'] = rawdata_ipc['sub ipc'].apply(lambda x: [regex.findall(xx)[0] for xx in x.split(';')])
        elif ipc_level == 3:
            data['main_ipc'] = rawdata_ipc['main ipc'].apply(lambda x: "".join(regex.findall(x)))
            data['sub_ipc'] = rawdata_ipc['sub ipc'].apply(lambda x: ["".join(regex.findall(xx)) for xx in x.split(';')])
        data['TC'+str(n_TC)] = rawdata_tc.apply(lambda x: x[np.arange(x['year']+1 if x['year']<2017 else 2017, x['year']+n_TC+1 if x['year']+n_TC<2018 else 2018).astype(str)].sum(), axis=1)
        data = data.set_index('number')
        seq_len = data['sub_ipc'].apply(lambda x: len(x)).max() + 3 # SOS - main ipc - sub ipcs - EOS

        main_ipcs = list(np.unique(data['main_ipc']))
        sub_ipcs = list(np.unique(np.concatenate(list(data['sub_ipc'].values))))
        all_ipcs = list(np.union1d(main_ipcs, sub_ipcs))

        vocab_w2i = {all_ipcs[i]: i for i in range(len(all_ipcs))}
        vocab_w2i.update({self.tokens[i]: len(all_ipcs)+i for i in range(len(self.tokens))})
        vocab_i2w = {i: all_ipcs[i] for i in range(len(all_ipcs))}
        vocab_i2w.update({len(all_ipcs)+i: self.tokens[i] for i in range(len(self.tokens))})
        vocab_size = len(vocab_w2i)

        return (data, vocab_w2i, vocab_i2w, vocab_size, seq_len)

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
        self.oversampled = oversampled
        self.labels = dataset.Y
        if self.oversampled:
            self.oversampled_idx = np.intersect1d(dataset.oversampled_idx, dataset.resampled_idx)
            self.original_idx = np.intersect1d(dataset.original_idx, dataset.resampled_idx)
            # self.labels_org = self.labels.loc[self.original_idx]
        else:
            self.original_idx = dataset.original_idx
            self.oversampled_idx = dataset.oversampled_idx

        self.dataset = dataset
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.n_folds = n_folds
        self.random_state = random_state

        self.idx_dict = {}
        self.split()

    def split(self):
        if self.stratify:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
            for train_idx, test_idx in splitter.split(np.zeros(len(self.labels)), self.labels):
                self.train_samples_idx = np.random.permutation(np.union1d(self.original_idx[train_idx], self.oversampled_idx))
                self.test_samples_idx = self.original_idx[test_idx]
            if self.n_folds == 1:
                self.idx_dict[0] = {'train': self.train_samples_idx, 'test': self.test_samples_idx}
            else:
                kf_splitter = StratifiedKFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
                fold = 0
                for train_idx, val_idx in kf_splitter.split(np.zeros(len(self.train_samples_idx)), self.labels[self.train_samples_idx]):
                    self.idx_dict[fold] = {'train': self.train_samples_idx[train_idx], 'val': self.train_samples_idx[val_idx], 'test': self.test_samples_idx}
                    fold += 1
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
            for train_idx, test_idx in splitter.split(np.zeros(len(self.dataset.data))):
                self.train_samples_idx = train_idx
                self.val_index = int(len(train_idx)*(1-self.val_ratio))
                self.test_samples_idx = test_idx
            if self.n_folds == 1:
                self.idx_dict[0] = {'train': self.train_samples_idx[:self.val_index], 'val': self.train_samples_idx[self.val_index:], 'test': self.test_samples_idx}
            else:
                kf_splitter = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
                fold = 0
                for train_idx, val_idx in kf_splitter.split(np.zeros(len(self.train_samples_idx)), self.labels[self.train_samples_idx]):
                    self.idx_dict[fold] = {'train': self.train_samples_idx[train_idx], 'val': self.train_samples_idx[val_idx], 'test': self.test_samples_idx}
                    fold += 1

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
