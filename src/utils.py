# Notes
'''
Author: Gyumin Lee
Version: 1.0
Description (primary changes): Last version before integration into master branch
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/Transformer/Tech_Gen/'
import sys
sys.path.append(root_dir)

# Basic libraries
import os
import copy
import pandas as pd
import numpy as np
import json
from collections.abc import Iterable

# DL libraries
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import sklearn
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def token2class(sequences, vocabulary=None, remove_extra=True):
    TOKEN_SOS = "<SOS>"
    TOKEN_PAD = "<PAD>"
    TOKEN_EOS = "<EOS>"

    assert vocabulary is not None, "Vocabulary is empty"
    if not isinstance(sequences, list):
        sequences = [sequences]
    if not isinstance(sequences[0], list):
        sequences = [sequences]

    outputs = []
    for sequence in sequences:
        output = [vocabulary[token] for token in sequence]
        if remove_extra:
            if TOKEN_EOS in output:
                output = output[:output.index(TOKEN_EOS)]
            elif TOKEN_PAD in output:
                temp_unq = np.unique(output, return_index=True)
                output = output[:temp_unq[1][list(temp_unq[0]).index(TOKEN_PAD)]]
            else:
                output = output[:]
        _, unq_idx = np.unique(output, return_index=True)
        output = list(np.array(output)[np.sort(unq_idx)])
        outputs.append(output)

    return outputs

def perturbed_decode(model, enc_inputs, perturb_degree=1e-2, configs={}):
    enc_inputs = enc_inputs.to(configs.model.device)
    enc_outputs, *_ = model.module.encoder(enc_inputs)

    enc_outputs_perturbed = torch.rand(enc_outputs.shape).to(device) * perturb_degree

    preds_recon = torch.tile(torch.tensor(configs.model.vocabulary["<SOS>"]), dims=(enc_inputs.shape[0], 1)).to(configs.model.device)
    preds_recon_perturbed = torch.tile(torch.tensor(configs.model.vocabulary["<SOS>"]), dims=(enc_inputs.shape[0], 1)).to(configs.model.device)

    for i in range(configs.model.n_dec_seq):
        with torch.no_grad():
            dec_outputs, *_ = model.module.decoder(preds_recon, enc_inputs, enc_outputs)
            dec_outputs_perturbed, *_ = model.module.decoder(preds_recon_perturbed, enc_inputs, enc_outputs_perturbed)
        pred_tokens = dec_outputs.argmax(2)[:,-1].unsqueeze(1)
        pred_tokens_perturbed = dec_outputs_perturbed.argmax(2)[:,-1].unsqueeze(1)
        preds_recon = torch.cat([preds_recon, pred_tokens], axis=1)
        preds_recon_perturbed = torch.cat([preds_recon_perturbed, pred_tokens_perturbed], axis=1)

    return preds_recon, preds_recon_perturbed

# reference: https://mws.readthedocs.io/en/develop/_modules/mws/utils/collections.html#DotDict
from collections.abc import Iterable, Mapping

def print_gpu_memcheck(verbose, devices=[], stage="Non-specified"):
    if verbose:
        usages = ", ".join([f"[cuda{str(device.index)}]{str(np.round(torch.cuda.memory_allocated(device)/1024/1024, 1))}Mb" for device in devices])
        print(f"[GPU-MEMCHECK] {stage}: {usages}")

def unique_list_order_preserved(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self)
        self.update(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            # No key by that name? Let's try being helpful.
            if f"@{name}" in self:
                # Does this same name occur starting with ``@``?
                return self[f"@{name}"]
            if f"#{name}" in self:
                # Does this same name occur starting with ``#``?
                return self[f"#{name}"]
            # Otherwise, raise the original exception
            raise AttributeError(name)

    def __setattr__(self, name, val):
        self.__setitem__(name, val)

    def __delattr__(self, name):
        self.__delitem__(name)

    def __setitem__(self, key, val):
        val = self.__class__.build(val)
        dict.__setitem__(self, key, val)

    def __iter__(self):
        return iter([self])

    def update(self, *args, **kwargs):
        for key, val in dict(*args, **kwargs).items():
            self[key] = self.__class__.build(val)

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            raw_dict = json.loads(f.read())
            return DotDict(raw_dict)

    @classmethod
    def build(cls, obj):
        if isinstance(obj, Mapping):
            # Return a new DotDict object wrapping `obj`.
            return cls(obj)
        if not isinstance(obj, str) and isinstance(obj, Iterable):
            # Build each item in the `obj` sequence,
            # then construct a new sequence matching `obj`'s type.
            # Must be careful not to pass strings here, even though they are iterable!
            return obj.__class__(cls.build(x) for x in obj)
        # In all other cases, return `obj` unchanged.
        return obj

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)
