# Notes
'''
Author: Gyumin Lee
Version: 0.1
Description (primary changes): Util functions
'''

# Set root directory
root_dir = '/home2/glee/Tech_Gen/'
import sys
sys.path.append(root_dir)

# Basic libraries
import os
import copy
import pandas as pd
import numpy as np

# DL libraries
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import sklearn
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_PAD = '<PAD>'

def token2class(sequences, vocabulary=None, remove_extra=True):
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
                output = output[1:output.index(TOKEN_EOS)]
            elif TOKEN_PAD in output:
                temp_unq = np.unique(output, return_index=True)
                output = output[1:temp_unq[1][list(temp_unq[0]).index(TOKEN_PAD)]]
            else:
                output = output[1:]
        _, unq_idx = np.unique(output, return_index=True)
        output = list(np.array(output)[np.sort(unq_idx)])
        outputs.append(output)

    return outputs
