# Notes
'''
Author: Gyumin Lee
Version: 1.2
Description (primary changes): Claim + class -> class
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/class2class/Tech_Gen/'
import sys
sys.path.append(root_dir)

# Basic libraries
import os
import re
import pandas as pd
import numpy as np

# DL libraries
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit, KFold
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
from transformers import DistilBertTokenizer
from transformers import T5Tokenizer

# Text cleaning libraries
from nltk.corpus import stopwords
from cleantext.sklearn import CleanTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extract_class_pattern = re.compile(
    r'([A-Za-z])'      # 1) 첫글자
    r'(\d{2})'         # 2) 두 자리 숫자
    r'([A-Za-z])'      # 3) 두 번째 문자
    r'(\d{0,3})'       # 4) pre-슬래시 숫자 (1~3자리)
    r'/'
    r'(\d{0,6})'       # 5) post-슬래시 숫자 (1~6자리)
)

def extract_class_level(code, level):
    if not isinstance(code, str):
        return []
    out = []
    for m in extract_class_pattern.finditer(code):
        L1, D2, L2, pre, post = m.groups()
        pre3  = pre.zfill(3)
        post6 = post.zfill(6)
        if level == 1: # Section-Subsection (e.g., "A61")
            out.append(f"{L1}{D2}")
        elif level == 2: # Section-Subsection-Class (e.g., "A61K")
            out.append(f"{L1}{D2}{L2}")
        elif level == 3: # Section-Subsection-Class-Main group (e.g., "A61K03")
            out.append(f"{L1}{D2}{L2}{pre3}")
        elif level == 4: # Section-Subsection-Class-Main group-/-Sub group (e.g., "A61K03/45")
            out.append(f"{L1}{D2}{L2}{pre3}/{post6}")
        else:
            raise ValueError(f"Not implemented for an PC level {level}")
    return out

class TechDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

        try:
            self.rawdata = pd.read_csv(os.path.join(config.data_dir, config.data_file), low_memory=False, nrows=self.data_nrows)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {os.path.join(config.data_dir, config.data_file)}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file at {os.path.join(config.data_dir, config.data_file)} is empty or invalid.")
        self.target_classes = None

        # Validate required columns in rawdata
        required_columns = ['main_cpc', 'sub_cpc', 'claims', 'patent_number']
        missing_columns = [col for col in required_columns if col not in self.rawdata.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing in the rawdata: {missing_columns}")
        self.data = self.preprocess()
        self.tokenizers = self.get_tokenizers()
        self.original_idx = np.array(self.data.index)
        self.X, self.Y, self.Y_digitized = self.make_io()
        if isinstance(self.X, tuple):
            self.X_claim, self.X_class = self.X
        self.n_outputs = len(np.unique(self.Y)) if self.pred_type == "classification" else 1

    def preprocess(self):
        regex = re.compile("[0-9a-zA-Z\/]+")
#         latest_year = datetime.datetime.now().year-1
        latest_year = 2022
        cols_year = ['<1976']+list(np.arange(1976,latest_year).astype(str))

        # IPC -> CPC 변경, data 내부에서 특허 클래스 지칭하는 용어를 pc로 변경
        if self.class_system == "CPC":
            name_main_class, name_sub_class = "main_cpc", "sub_cpc"
            rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main_cpc', 'sub_cpc', 'claims'])[['patent_number','main_cpc','sub_cpc','claims']]
        elif self.class_system == "IPC":
            name_main_class, name_sub_class = "main_ipc", "sub_ipc"
            rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main_ipc', 'sub_ipc', 'claims'])[['patent_number','main_ipc','sub_ipc','claims']]
            
        if len(rawdata_dropna) == 0 and self.config.data_nrows is not None:
            print("No row in rawdata has ipc -> reload rawdata")
            while len(rawdata_dropna) == 0:
                self.rawdata = pd.read_csv(os.path.join(self.config.data_dir, self.config.data_file), low_memory=False, skiprows=lambda idx: idx != 0 and np.random.random() > 0.01)
                rawdata_dropna = self.rawdata.dropna(axis=0, subset=['main_ipc', 'sub_ipc', 'claims'])[['patent_number','main_ipc','sub_ipc','claims']]
            
        main_pcs = [x for x in pd.unique(rawdata_dropna[name_main_class])]        
        assert len(main_pcs) != 0, "target patent class is not observed"

        if self.data_type == "class" or self.data_type in ["class+claim", "claim+class"]:
            data = rawdata_dropna[["patent_number"]].copy(deep=True)
            assert self.class_level in [1,2,3,4], f"Not implemented for an PC level {self.class_level}"
            
            # for IPC, formatting
            if self.class_system == "IPC":
                rawdata_dropna.loc[:, name_main_class] = rawdata_dropna[name_main_class].apply(lambda x: ";".join([s.split()[0] for s in re.findall(r'"([^"]*)"', x)]))
                rawdata_dropna.loc[:, name_sub_class] = rawdata_dropna[name_sub_class].apply(lambda x: ";".join([s.split()[0] for s in re.findall(r'"([^"]*)"', x)]))
            
            data["main_class"] = rawdata_dropna[name_main_class].apply(lambda x: [] if x is None else extract_class_level(x, self.class_level))
            data["sub_class"] = rawdata_dropna[name_sub_class].apply(lambda x: [] if x is None else extract_class_level(x, self.class_level))
            data["patent_classes"] = data.apply(lambda x: x["main_class"] + x["sub_class"], axis=1)

            seq_len = 1 + data['main_class'].apply(lambda x: len(x)).max() + data['sub_class'].apply(lambda x: len(x)).max() + 1 # SOS - main ipc - sub ipcs - EOS
            self.max_seq_len_class = seq_len if self.max_seq_len_class < seq_len else self.max_seq_len_class
            if self.data_type in ["class+claim", "claim+class"]:
                data["claims"] = rawdata_dropna.loc[data.index]["claims"]
        elif self.data_type == "claim":
            data = rawdata_dropna.loc[rawdata_dropna[name_main_class].isin(main_pcs)]

        rawdata_tc = self.rawdata.loc[data.index]

        ## Get number of forward citations within 5 years (TC5)
        data["TC"+str(self.n_TC)] = rawdata_tc.apply(lambda x: x[np.arange(x["granted_year"] if x["granted_year"]<latest_year else latest_year, x["granted_year"]+1+self.n_TC if x["granted_year"]+1+self.n_TC<latest_year+1 else latest_year).astype(str)].sum(), axis=1)
        data = data.reset_index(drop=True)

        ## Get digitized number of forward citations within 5 years (TC5_digitized)
        bins_criterion = [data["TC"+str(self.n_TC)].quantile(0.9)]
        data["TC"+str(self.n_TC)+"_digitized"] = pd.Series(np.digitize(data["TC"+str(self.n_TC)].values, bins=bins_criterion, right=True))

        ## Get patent class
        patent_classes = data["main_class"].apply(lambda x: x[0]).values
        label_encoder = LabelEncoder()
        label_encoder.fit(patent_classes)
        data["class"] = pd.Series(label_encoder.fit_transform(patent_classes))

        if self.pred_target == "class":
            self.target_classes = label_encoder.classes_
        elif self.pred_target == "citation":
            self.target_classes = ["Least_valuable", "Most_valuable"]

        data.index = pd.Index(data["patent_number"].apply(lambda x: str(x) if not isinstance(x, str) else x))
#         data = data.set_index("patent_number", drop=False)

        return data

    def train_custom_tokenizer(self):
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
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>", length=self.max_seq_len_claim)
        if self.max_seq_len_claim > 0:
            tokenizer.enable_truncation(max_length=self.max_seq_len_claim)
        tokenizer.vocab_size = tokenizer.get_vocab_size()
        tokenizer.decoder = decoders.WordPiece()

        print("Tokenizer is trained and saved")

        return tokenizer

    def get_tokenizers(self):
        def custom_token_to_id(self): return self.convert_tokens_to_ids
        def custom_decode_batch(self): return self.batch_decode
        def custom_encode_batch(self): return self.encode
        if self.data_type == "claim":
            if self.pretrained_enc:
                enc_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                enc_tokenizer.token_to_id = custom_token_to_id(enc_tokenizer)
                enc_tokenizer.decode_batch = custom_decode_batch(enc_tokenizer)
                enc_tokenizer.encode_batch = custom_encode_batch(enc_tokenizer)
            else:
                enc_tokenizer = self.train_custom_tokenizer()

            if self.pretrained_dec:
                dec_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                dec_tokenizer.token_to_id = custom_token_to_id(dec_tokenizer)
                dec_tokenizer.decode_batch = custom_decode_batch(dec_tokenizer)
                dec_tokenizer.encode_batch = custom_encode_batch(dec_tokenizer)
            else:
                if self.pretrained_enc:
                    dec_tokenizer = enc_tokenizer
                else:
                    dec_tokenizer = self.train_custom_tokenizer()
            tokenizers = {"enc": enc_tokenizer, "dec": dec_tokenizer}
        elif self.data_type == "class":
            enc_tokenizer = PatentClassTokenizer(self.data, max_len=self.max_seq_len_class)
            dec_tokenizer = enc_tokenizer
            tokenizers = {"enc": enc_tokenizer, "dec": dec_tokenizer}
        elif self.data_type in ["class+claim", "claim+class"]:
            if self.pretrained_enc:
                claim_enc_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                claim_enc_tokenizer.token_to_id = custom_token_to_id(claim_enc_tokenizer)
                claim_enc_tokenizer.decode_batch = custom_decode_batch(claim_enc_tokenizer)
                claim_enc_tokenizer.encode_batch = custom_encode_batch(claim_enc_tokenizer)
            else:
                claim_enc_tokenizer = self.train_custom_tokenizer()
            claim_dec_tokenizer = claim_enc_tokenizer

            class_enc_tokenizer = PatentClassTokenizer(self.data, max_len=self.max_seq_len_class)
            class_dec_tokenizer = class_enc_tokenizer
            tokenizers = {"claim_enc": claim_enc_tokenizer, "claim_dec": claim_dec_tokenizer, "class_enc": class_enc_tokenizer, "class_dec": class_dec_tokenizer}

        return tokenizers

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
        if self.data_type == "class":
            X = self.data["patent_classes"]
        elif self.data_type == "claim":
            claim_separator = "\n\n\n"
            if self.claim_level != -1:
                X = self.data["claims"].apply(lambda x: "".join(x.split(claim_separator)[:self.claim_level]) if len(x.split(claim_separator))>=self.claim_level else x)
            else:
                X = self.data["claims"]
        elif self.data_type in ["class+claim", "claim+class"]:
            X_class = self.data["patent_classes"]
            claim_separator = "\n\n\n"
            if self.claim_level != -1:
                X_claim = self.data["claims"].apply(lambda x: "".join(x.split(claim_separator)[:self.claim_level]) if len(x.split(claim_separator))>=self.claim_level else x)
            else:
                X_claim = self.data["claims"]
            X = (X_claim, X_class)

        if self.pred_target == "citation":
            Y_digitized = self.data["TC"+str(self.n_TC)+"_digitized"]
            if self.pred_type == "regression":
                Y = self.data["TC"+str(self.n_TC)]
            elif self.pred_type == "classification":
                Y = self.data["TC"+str(self.n_TC)+"_digitized"]
        elif self.pred_target == "class":
            Y = Y_digitized = self.data["class"]
        else:
            Y = None

        return X, Y, Y_digitized

    def tokenize(self, tokenizer, texts):
        if str(tokenizer.__class__).split("\'")[1].split(".")[0] == "transformers": # Tokenizer for pretrained language models
            tokenized = tokenizer(texts, add_special_tokens=True, max_length=self.max_seq_len_claim, padding="max_length", truncation=True)
            tokenized_dict = {"input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                                 "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long)}
            # tokenized_dict = {"input_ids": tokenized["input_ids"].to(dtype=torch.long),
            #                     "attention_mask": tokenized["attention_mask"].to(dtype=torch.long)}
        elif str(tokenizer.__class__).split("\'")[1].split(".")[0] == "tokenizers": # Custom tokenizer
            tokenized = tokenizer.encode(texts) if isinstance(texts, str) else tokenizer.encode_batch(texts)
            tokenized_dict = {"input_ids": torch.tensor(tokenized.ids, dtype=torch.long),
                                 "attention_mask": torch.tensor(tokenized.attention_mask, dtype=torch.long)}
        return tokenized_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_dtype = torch.long if self.pred_type=="classification" else torch.float32
        if self.Y is None or len(self.Y) == 0:
            raise ValueError("The target variable 'self.Y' is not initialized or is empty.")
        target_outputs = torch.tensor(self.Y[idx], dtype=target_dtype)

        if self.data_type == "class":
            input_classes = self.X[idx]
            output_claims = input_classes
            text_inputs_dict = torch.tensor(self.tokenizers["enc"].encode(input_classes), dtype=torch.long)
            text_outputs_dict = text_inputs_dict
            out = {"text_inputs": text_inputs_dict, "text_outputs": text_outputs_dict, "targets": target_outputs}
        elif self.data_type == "claim":
            input_claims = self.extract_keywords([self.X[idx]])[0] if self.use_keywords else self.X[idx]
            output_claims = self.X[idx]
            text_inputs_dict = self.tokenize(self.tokenizers["enc"], input_claims)
            text_outputs_dict = self.tokenize(self.tokenizers["dec"], output_claims)
            out = {"text_inputs": text_inputs_dict, "text_outputs": text_outputs_dict, "targets": target_outputs}
        elif self.data_type in ["class+claim", "claim+class"]:
            input_claims = self.extract_keywords([self.X_claim[idx]])[0] if self.use_keywords else self.X_claim[idx]
            output_claims = self.X_claim[idx]
            claim_inputs_dict = self.tokenize(self.tokenizers["claim_enc"], input_claims)
            claim_outputs_dict = self.tokenize(self.tokenizers["claim_dec"], output_claims)
            input_classes = self.X_class[idx]
            output_classes = input_classes
            class_inputs_dict = torch.tensor(self.tokenizers["class_enc"].encode(input_classes), dtype=torch.long)
            class_outputs_dict = torch.tensor(self.tokenizers["class_dec"].encode(output_classes), dtype=torch.long)
            # out = {"text_inputs": {"claim_inputs": claim_inputs_dict, "class_inputs": class_inputs_dict}, "text_outputs": {"claim_outputs": claim_outputs_dict, "class_outputs": class_outputs_dict}, "targets": target_outputs}
            out = {"text_inputs": {"claim": claim_inputs_dict, "class": class_inputs_dict}, "text_outputs": class_outputs_dict, "targets": target_outputs}

        return out

class PatentClassTokenizer():
    def __init__(self, pc_data, max_len=100):
        self.pcs = pc_data["patent_classes"].values
        self.max_len = max_len
        self.sos_token = "<SOS>"
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.set_vocabulary()
        self.sos_id = self.vocab_w2i[self.sos_token]
        self.pad_id = self.vocab_w2i[self.pad_token]
        self.eos_id = self.vocab_w2i[self.eos_token]
        self.unk_id = self.vocab_w2i[self.unk_token]

    def set_vocabulary(self):
        self.vocabulary = np.unique(np.concatenate(self.pcs))
        self.special_tokens = [self.sos_token, self.pad_token, self.eos_token, self.unk_token]
        self.vocab_w2i = {self.special_tokens[i]: i for i in range(len(self.special_tokens))}
        self.vocab_w2i.update({self.vocabulary[i]: len(self.special_tokens)+i for i in range(len(self.vocabulary))})
        self.vocab_i2w = {i: self.special_tokens[i] for i in range(len(self.special_tokens))}
        self.vocab_i2w.update({len(self.special_tokens)+i: self.vocabulary[i] for i in range(len(self.vocabulary))})
        self.vocab_size = len(self.vocabulary) + len(self.special_tokens)

    def token_to_id(self, token):
        try:
            out = self.vocab_w2i[token]
        except:
            out = self.vocab_w2i[self.unk_token]
        return out

    def encode(self, tokens):
        # if not isinstance(tokens, list): tokens = [tokens]
        if isinstance(tokens, int) or isinstance(tokens, str): tokens = [tokens]
        out = [self.token_to_id(self.sos_token)] + [self.token_to_id(token) for token in tokens]
        if len(out) < self.max_len:
            out = out + [self.token_to_id(self.eos_token)] + [self.token_to_id(self.pad_token) for _ in range(self.max_len - len(out) - 1)]
        return out

    def id_to_token(self, id):
        try:
            out = self.vocab_i2w[id]
        except:
            out = self.unk_token
        return out

    def decode(self, ids):
        # if not isinstance(ids, list) or not isinstance(ids, np.ndarray): ids = [ids]
        if isinstance(ids, int) or isinstance(ids, str): ids = [ids]
        out = [self.id_to_token(id) for id in ids]
        return out

    def get_vocab_size(self):
        return self.vocab_size

    def encode_batch(self, batch):
        return [self.encode(row) for row in batch]

    def decode_batch(self, batch):
        return [self.decode(row) for row in batch]

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
