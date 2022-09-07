# Notes
'''
Author: Gyumin Lee
Version: 0.2
Description (primary changes): Add attention decoder
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
    def __init__(self, embedding_dim, vocab_size, hidden_dim, n_layers, device, dropout=0.1, padding_idx=None):
        super(Encoder_SEQ, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.vocab_size = vocab_size

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True).to(device)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, inputs):
        embedded = self.dropout(self.embedding(inputs))
        hidden = self.initHidden(len(inputs)).to(self.device)
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

class Decoder_SEQ(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, n_layers, device, dropout=0.1):
        super(Decoder_SEQ, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.vocab_size = vocab_size

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True).to(device)
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.linear = nn.Linear(hidden_dim, vocab_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, inputs, hidden=None):
        if hidden == None:
            hidden = self.initHidden(len(inputs))
        # inputs = inputs.unsqueeze(0)
        # n_batch = inputs.shape[1]
        embedded = self.dropout(self.embedding(inputs))
        # embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden)
        pred = self.linear(output.squeeze(1))
        pred = self.softmax(pred)

        return pred, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

class AttnDecoder_SEQ(Decoder_SEQ):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, n_layers, device, max_len=99, dropout=0.1):
        super(AttnDecoder_SEQ, self).__init__(embedding_dim, vocab_size, hidden_dim, n_layers, device)
        self.max_len = max_len
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True).to(device)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_len)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, inputs, hidden, encoder_outputs):
        if hidden is None:
            hidden = self.initHidden(len(inputs))
        embedded = self.dropout(self.embedding(inputs))

        attn_weights = F.softmax(self.attn(torch.cat((embedded[:,0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        output = torch.cat((embedded[:,0], attn_applied[:,0]), 1)
        output = F.relu(self.attn_combine(output).unsqueeze(1))

        output, hidden = self.gru(output, hidden)
        pred = self.linear(output.squeeze(1))
        pred = self.softmax(pred)

        return pred, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

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
        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
        "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
        "Encoder and decoder must have equal number of layers!"

    def forward(self, x, teach_force_ratio=0.75):
        batch_size = x.size(0)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, vocab_size, self.max_len).to(device=self.device)
        outputs[:,0,0] = 1.

        # encoder_outputs = torch.zeros(self.max_len, self.encoder.hidden_dim, device=self.device)
        encoder_output, z = self.encoder(x)
        #
        # for ei in range(self.max_len):
        #     enco

        next_hidden = z
        next_input = torch.from_numpy(np.tile([self.dataset.vocab_w2i[TOKEN_SOS]], batch_size)).unsqueeze(1).to(device=self.device)
        for t in range(1, self.max_len):
            output, next_hidden, pred_token = self.pred_next(next_input, next_hidden, encoder_output)

            rand_num = np.random.random()
            if rand_num < teach_force_ratio:
                next_input = x[:,t].unsqueeze(1)
            else:
                next_input = pred_token.unsqueeze(1)
            outputs[:,:,t] = output

        return outputs, z

    def pred_next(self, _next_input, _next_hidden, _encoder_outputs):
        _output, _hidden, _attn_weights = self.decoder(_next_input, _next_hidden, _encoder_outputs)
        _pred_token = _output.argmax(1)
        if _output.size(0) != 1: _pred_token = _pred_token.squeeze(0)

        if _pred_token.size() == torch.Size([]):
            print(_output.shape, _hidden.shape, _pred_token.shape, _next_input.shape)

        if self.dataset.vocab_w2i[TOKEN_EOS] in _next_input.view(-1):
            # print("ERROR",_next_input==self.dataset.vocab_w2i[TOKEN_EOS])
            _pred_token[_next_input.view(-1)==self.dataset.vocab_w2i[TOKEN_EOS]] = self.dataset.vocab_w2i[TOKEN_PAD]

        return _output, _hidden, _pred_token

class SEQ2SEQ_pred(nn.Module):
    def __init__(self, enc_c, pred, enc_t, device):
        super(SEQ2SEQ_pred, self).__init__()
        self.compound_encoder = enc_c
        self.predictor = pred
        self.target_encoder = enc_t
        self.device = device

    def forward(self, compound, target):
        batch_size = compound.size(0)
        compound_len = compound.size(1)

        z_c = self.compound_encoder(compound)
        z_t = self.target_encoder(target)
        z_i = torch.cat((z_c, z_t), dim=2)

        output = self.predictor(z_i)

        return output, z_c, z_t, z_i

class Predictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(latent_dim, hidden_dim).float()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).float()
        self.out = nn.Linear(hidden_dim, output_dim).float()

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        hidden = F.relu(self.linear2(hidden))
        #hidden = self.linear2(hidden)
        pred = F.relu(self.out(hidden))

        return pred
