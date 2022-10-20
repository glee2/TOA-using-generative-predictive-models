# Notes
'''
Author: Gyumin Lee
Version: 0.3
Description (primary changes): Add BatchNorm
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
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset

TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_PAD = '<PAD>'

class Encoder_SEQ(nn.Module):
    def __init__(self, device, params={}):
        super(Encoder_SEQ, self).__init__()
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.n_layers = params['n_layers']
        self.vocab_size = params['vocab_size']
        self.bidirec = params['bidirec']
        self.n_directions = params['n_directions']
        self.padding_idx = params['padding_idx']
        self.dropout = params['dropout']
        self.device = device

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirec)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim * self.n_directions, self.hidden_dim)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len)
        batch_size = inputs.shape[0]

        embedded = self.dropout(self.embedding(inputs)) # (batch_size, seq_len, embedding_dim)
        hidden = self.initHidden(batch_size) # (n_layers * n_directions, batch_size, hidden_dim)
        output, hidden = self.gru(embedded, hidden)
        # output: (batch_size, seq_len, hidden_dim * n_directions), hidden: (n_layers * n_directions, batch_size, hidden_dim)

        hidden = hidden.view(self.n_layers, batch_size, self.hidden_dim * self.n_directions) # Separate layer and direction
        # hidden: (n_layers, batch_size, hidden_dim * n_directions)

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros((self.n_layers * self.n_directions, batch_size, self.hidden_dim), device=self.device)

class Attention(nn.Module):
    def __init__(self, device, params={}):
        super(Attention, self).__init__()
        self.hidden_dim = params['hidden_dim']
        self.n_directions = params['n_directions']
        self.device = device

        self.attn = nn.Linear((self.hidden_dim * self.n_directions) + (self.hidden_dim * self.n_directions), self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (n_layers, batch_size, hidden_dim * n_directions), encoder_outputs: (batch_size, seq_len, hidden_dim * n_directions)

        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden_last = hidden[-1] # (batch_size, hidden_dim * n_directions)

        hidden = hidden_last.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_dim * n_directions)
        # hidden = hidden.repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) # (batch_size, seq_len, hidden_dim)
        attention = self.v(energy).squeeze(2) # (batch_size, seq_len)

        return F.softmax(attention, dim=1)

class AttnDecoder_SEQ(nn.Module):
    def __init__(self, device, attention, params={}):
        super().__init__()
        self.embedding_dim = params['embedding_dim']
        self.vocab_size = params['vocab_size']
        self.hidden_dim = params['hidden_dim']
        self.n_layers = params['n_layers']
        self.max_len = params['max_len']
        self.n_directions = params['n_directions']
        self.dropout = params['dropout']
        self.device = device

        self.attention = attention
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU((self.hidden_dim * self.n_directions) + self.embedding_dim, (self.hidden_dim * self.n_directions), self.n_layers, batch_first=True)
        self.fc_out = nn.Linear(self.embedding_dim + (self.hidden_dim * self.n_directions) + (self.hidden_dim * self.n_directions), self.vocab_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, inputs, hidden, encoder_outputs):
        # inputs: (batch_size), hidden: (n_layers, batch_size, hidden_dim * n_directions), encoder_outputs: (batch_size, seq_len, hidden_dim * n_directions)
        inputs = inputs.unsqueeze(1) # (batch_size, 1)

        embedded = self.dropout(self.embedding(inputs)) # (batch_size, 1, embedding_dim)

        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1) # (batch_size, 1, seq_len)

        weighted = torch.bmm(a, encoder_outputs) # (batch_size, 1, hidden_dim * n_directions)

        gru_input = torch.cat((embedded, weighted), dim=2) # (batch_size, 1, hidden_dim * n_directions + embedding_dim)

        output, hidden = self.gru(gru_input, hidden) # output: (batch_size, 1, hidden_dim * n_directions), hidden: (n_layers, batch_size, hidden_dim * n_directions)

        embedded = embedded.squeeze(1)
        weighted = weighted.squeeze(1)
        output = output.squeeze(1)

        prediction = self.fc_out(torch.cat((embedded, weighted, output), dim=1)) # (batch_size, vocab_size)

        return prediction, hidden

class Predictor(nn.Module):
    def __init__(self, device, params={}):
        super(Predictor, self).__init__()
        self.latent_dim = params['latent_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim_predictor']
        self.dropout = params['dropout']
        self.device = device

        self.linear = nn.Linear(self.latent_dim, self.hidden_dim).float()
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim).float()
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.out = nn.Linear(self.hidden_dim, self.output_dim).float()

    def forward(self, x):
        hidden = F.relu(self.bn1(self.linear(x)))
        hidden = F.relu(self.bn2(self.linear2(hidden)))
        hidden = self.dropout(hidden)
        pred = self.out(hidden)

        return pred

class SEQ2SEQ(nn.Module):
    def __init__(self, device, enc=None, dec=None, pred=None, vocab={}, max_len=99, u=0.5):
        super(SEQ2SEQ, self).__init__()
        self.encoder = enc
        self.decoder = dec
        self.predictor = pred
        self.vocab = vocab
        self.max_len = max_len
        self.u = u
        self.device = device

    def forward(self, x, teach_force_ratio=0.75):
        # x: (batch_size, seq_len)
        batch_size = x.size(0)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros((batch_size, vocab_size, self.max_len), device=self.device) # (batch_size, vocab_size, seq_len)
        outputs[:,self.vocab[TOKEN_SOS],0] = 1.

        encoder_output, z = self.encoder(x)
        # encoder_output: (batch_size, seq_len, hidden_dim * n_directions), z: (n_layers, batch_size, hidden_dim * n_directions)

        next_hidden = z
        next_input = torch.tensor(np.tile([self.vocab[TOKEN_SOS]], batch_size), device=self.device) # (batch_size)
        for t in range(1, self.max_len):
            output, next_hidden, pred_token = self.pred_next(next_input, next_hidden, encoder_output)
            # output: (batch_size, vocab_size), hidden: (batch_size, hidden_dim * n_directions), pred_token: (batch_size)

            rand_num = np.random.random()
            if rand_num < teach_force_ratio:
                next_input = x[:,t] # (batch_size)
            else:
                next_input = pred_token # (batch_size)
            outputs[:,:,t] = output

        z_flattened = torch.permute(z, (1,0,2)).reshape(batch_size, -1) # (batch_size, hidden_dim * n_directions * n_layers)
        pred_y = self.predictor(z_flattened)

        return outputs, pred_y, z

    def pred_next(self, next_input, next_hidden, encoder_outputs):
        output, hidden = self.decoder(next_input, next_hidden, encoder_outputs)
        # output: (batch_size, vocab_size), hidden: (n_layers, batch_size, hidden_dim * n_directions)

        pred_token = output.argmax(1) # (batch_size)
        if output.size(0) != 1: pred_token = pred_token.squeeze(0)

        if pred_token.size() == torch.Size([]):
            print(output.shape, hidden.shape, pred_token.shape, next_input.shape)

        if self.vocab[TOKEN_EOS] in next_input.view(-1):
            pred_token[next_input.view(-1)==self.vocab[TOKEN_EOS]] = self.vocab[TOKEN_PAD]

        return output, hidden, pred_token
