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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_PAD = '<PAD>'

class Encoder_SEQ(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, n_layers, device, bidirec=False, dropout=0.1, padding_idx=None):
        super(Encoder_SEQ, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.vocab_size = vocab_size
        self.bidirec = bidirec
        self.n_directions = 2 if self.bidirec else 1
        # self.hidden_dim_enc = self.hidden_dim * self.n_directions if self.bidirec else self.hidden_dim

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirec).to(device)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.fc = nn.Linear(self.hidden_dim * self.n_directions, hidden_dim)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len)
        batch_size = inputs.shape[0]

        embedded = self.dropout(self.embedding(inputs)) # (batch_size, seq_len, embedding_dim)
        hidden = self.initHidden(len(inputs)).to(self.device) # (n_layers * n_directions, batch_size, hidden_dim)
        output, hidden = self.gru(embedded, hidden)
        # output: (batch_size, seq_len, hidden_dim * n_directions), hidden: (n_layers * n_directions, batch_size, hidden_dim)

        hidden = hidden.view(self.n_layers, batch_size, self.hidden_dim * self.n_directions) # Separate layer and direction
        # hidden: (n_layers, batch_size, hidden_dim * n_directions)

        # hidden = hidden.view(self.n_layers, self.n_directions, batch_size, self.hidden_dim) # Separate layer and direction
        # hidden = hidden[-1].view(batch_size, -1) # hidden weights from the last layer -> (batch_size, hidden_dim_enc)

        # if self.bidirec:
        #     hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))) # (batch_size, hidden_dim)
        # else:
        #     hidden = hidden[-1,:,:] # hidden weights from the last layer

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_dim).to(self.device)

class Attention(nn.Module):
    def __init__(self, hidden_dim, n_directions):
        super().__init__()

        self.attn = nn.Linear((hidden_dim * n_directions) + (hidden_dim * n_directions), hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

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
    def __init__(self, embedding_dim, vocab_size, hidden_dim, n_directions, attention, n_layers, device, max_len=99, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        # self.hidden_dim * n_directions = hidden_dim * n_directions
        self.n_layers = n_layers
        self.device = device
        self.max_len = max_len

        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU((hidden_dim * n_directions) + embedding_dim, (hidden_dim * n_directions), n_layers, batch_first=True).to(device)
        self.fc_out = nn.Linear(embedding_dim + (hidden_dim * n_directions) + (hidden_dim * n_directions), vocab_size)
        self.dropout = nn.Dropout(dropout)

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
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout=0.75):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(latent_dim, hidden_dim).float()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).float()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim).float()

    def forward(self, x):
        hidden = F.relu(self.bn1(self.linear(x)))
        hidden = F.relu(self.bn2(self.linear2(hidden)))
        hidden = self.dropout(hidden)
        pred = self.out(hidden)

        return pred

class SEQ2SEQ(nn.Module):
    def __init__(self, device='cpu', dataset=None, enc=None, dec=None, pred=None, max_len=99, u=0.5):
        super(SEQ2SEQ, self).__init__()
        self.device = device
        self.dataset = dataset
        self.encoder = enc
        self.decoder = dec
        self.predictor = pred
        self.max_len = max_len
        self.u = u

    def forward(self, x, teach_force_ratio=0.75):
        # x: (batch_size, seq_len)
        batch_size = x.size(0)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, vocab_size, self.max_len).to(device=self.device) # (batch_size, vocab_size, seq_len)
        outputs[:,self.dataset.vocab_w2i[TOKEN_SOS],0] = 1.

        # encoder_outputs = torch.zeros(self.max_len, self.encoder.hidden_dim, device=self.device)
        encoder_output, z = self.encoder(x)
        # encoder_output: (batch_size, seq_len, hidden_dim * n_directions), z: (n_layers, batch_size, hidden_dim * n_directions)

        next_hidden = z
        next_input = torch.from_numpy(np.tile([self.dataset.vocab_w2i[TOKEN_SOS]], batch_size)).to(device=self.device) # (batch_size)
        for t in range(1, self.max_len):
            output, next_hidden, pred_token = self.pred_next(next_input, next_hidden, encoder_output)
            # output: (batch_size, vocab_size), hidden: (batch_size, hidden_dim * n_directions), pred_token: (batch_size)

            rand_num = np.random.random()
            if rand_num < teach_force_ratio:
                # next_input = x[:,t].unsqueeze(1)
                next_input = x[:,t] # (batch_size)
            else:
                # next_input = pred_token.unsqueeze(1)
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

        if self.dataset.vocab_w2i[TOKEN_EOS] in next_input.view(-1):
            pred_token[next_input.view(-1)==self.dataset.vocab_w2i[TOKEN_EOS]] = self.dataset.vocab_w2i[TOKEN_PAD]

        return output, hidden, pred_token
