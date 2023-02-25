# Notes
'''
Author: Gyumin Lee
Version: 0.7
Description (primary changes): Use pre-trained Tokenizer
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
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing

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

        self.rawdata = pd.read_csv(os.path.join(config.data_dir, "collection_final_with_claims.csv"))
        self.data = self.preprocess()
        self.tokenizer = self.get_tokenizer()
        self.original_idx = np.array(self.data.index)
        self.X, self.Y, self.Y_digitized = self.make_io()
        self.n_outputs = len(np.unique(self.Y_digitized)) if self.pred_type == "classification" else 1

    def get_tokenizer(self):
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
            # tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            tokenizer = Tokenizer(WordPiece(unk_token="<UNK>"))
            # trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["<SOS>", "<PAD>", "<EOS>", "<UNK>"])
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
            tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>")
            if self.max_seq_len > 0:
                tokenizer.enable_truncation(max_length=self.max_seq_len)
            tokenizer.save(tokenizer_path)
            print("Tokenizer is trained and saved")

        tokenizer.decoder = decoders.WordPiece()

        return tokenizer

    def make_io(self, val_main=10, val_sub=1):
        X_df = pd.DataFrame(index=self.data.index)
        if self.data_type == "class":
            X_df['main'] = self.data['main_ipc'].apply(lambda x: self.vocab_w2i[x])
            X_df['sub'] = self.data['sub_ipc'].apply(lambda x: [self.vocab_w2i[xx] for xx in x])
            main_sub_combined = X_df.apply(lambda x: [x['main']]+x['sub'], axis=1)
            X = main_sub_combined.apply(lambda x: np.concatenate([[self.vocab_w2i[TOKEN_SOS]]+x+[self.vocab_w2i[TOKEN_EOS]], np.zeros(self.seq_len-(len(x)+2))+self.vocab_w2i[TOKEN_PAD]]).astype(int))
        elif self.data_type == "claim":
            tokenized_outputs = self.tokenizer.encode_batch(self.data['claims'])
            X = pd.Series([output.ids for output in tokenized_outputs])

        X = np.vstack(X.values)
        Y = self.data['TC'+str(self.n_TC)].values

        Y_digitized = np.digitize(Y, bins=[0], right=True)

        if self.pred_type == "classification":
            Y = Y_digitized

        return X, Y, Y_digitized

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
            claim_separator = "\n\n\n"
            rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main ipc','claims'])[['number','main ipc','claims']]
            if self.target_ipc == "ALL":
                main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc'])]
            else:
                main_ipcs = [x for x in pd.unique(rawdata_dropna['main ipc']) if self.target_ipc in x]
            assert len(main_ipcs) != 0, "target ipc is not observed"
            data = rawdata_dropna.loc[rawdata_dropna['main ipc'].isin(main_ipcs)]

            if self.claim_level != -1:
                data['claims'] = data['claims'].apply(lambda x: "".join(x.split(claim_separator)[:self.claim_level]) if len(x.split(claim_separator))>=self.claim_level else x)

        rawdata_tc = self.rawdata.loc[data.index][['year']+cols_year]
        data['TC'+str(self.n_TC)] = rawdata_tc.apply(lambda x: x[np.arange(x['year']+1 if x['year']<2017 else 2017, x['year']+self.n_TC+1 if x['year']+self.n_TC<2018 else 2018).astype(str)].sum(), axis=1)
        data = data.set_index('number')

        return data

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
