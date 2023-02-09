# Notes
'''
Author: Gyumin Lee
Version: 0.1
Description (primary changes): Util functions
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

'''
class DotDict(dict):
    this = "this"
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

        args_with_kwargs = []
        for arg in args:
            args_with_kwargs.append(arg)
        args_with_kwargs.append(kwargs)
        args = args_with_kwargs

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
                    if isinstance(v, dict):
                        self[k] = DotDict(v)
                    elif isinstance(v, str) or isinstance(v, bytes):
                        self[k] = v
                    elif isinstance(v, Iterable):
                        klass = type(v)
                        map_value: List[Any] = []
                        for e in v:
                            map_e = DotDict(e) if isinstance(e, dict) else e
                            map_value.append(map_e)
                        self[k] = klass(map_value)

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            raw_dict = json.loads(f.read())
            return DotDict(raw_dict)

    @classmethod
    def update(cls, d):
        # return cls.__dict__.update(d)
        print(type(cls), cls)
        print(cls.this)
        return cls

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
'''

# reference: https://mws.readthedocs.io/en/develop/_modules/mws/utils/collections.html#DotDict
from collections.abc import Iterable, Mapping

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
