# Notes
'''
Author: Gyumin Lee
Version: 0.5
Description (primary changes): Add Transformer
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
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset

## weight initialization
def init_weights(m):
    if not isinstance(m, nn.Module): return

    for module in m.named_modules():
        if "Linear" in module[1].__class__.__name__:
            torch.nn.init.xavier_uniform_(module[1].weight)
        elif "GRU" in module[1].__class__.__name__:
            if hasattr(m, "n_layers") and hasattr(m, "n_directions"):
                for i in range(m.n_layers * m.n_directions):
                    for j in range(len(module[1]._all_weights[i])):
                        if "weight" in module[1]._all_weights[i][j]:
                            torch.nn.init.xavier_uniform_(module[1].all_weights[i][j])
                        elif "bias" in module[1]._all_weights[i][j]:
                            torch.nn.init.uniform_(module[1].all_weights[i][j])

## positional encoding
def get_sinusoid_encoding_table(n_seq, d_hidden):
    def cal_angle(position, i_hidden):
        return position / np.power(10000, 2 * (i_hidden // 2) / d_hidden)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidden) for i_hidden in range(d_hidden)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos
    return sinusoid_table

## Padding index masking
def get_pad_mask(seq_q, seq_k, i_padding):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_mask = seq_k.data.eq(i_padding)
    pad_mask = pad_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_mask

## Subsequent word masking
def get_subsequent_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.dropout = nn.Dropout(self.config.p_dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        # Q: (batch_size, n_head, n_q_seq, d_head), K: (batch_size, n_head, n_k_seq, d_head), V: (batch_size, n_head, n_v_seq, d_head)

        scores = torch.matmul(Q, K.transpose(-1, -2)) # scores: (batch_size, n_head, n_q_seq, n_k_seq)
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)

        attn_prob = nn.Softmax(dim=-1)(scores) # attn_prob: (batch_size, n_head, n_q_seq, n_k_seq)
        attn_prob = self.dropout(attn_prob)

        context = torch.matmul(attn_prob, V) # context: (batch_size, n_head, n_q_seq, d_head)

        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device

        self.W_Q = nn.Linear(self.config.d_hidden, self.config.n_head * self.config.d_head).to(self.device)
        self.W_K = nn.Linear(self.config.d_hidden, self.config.n_head * self.config.d_head).to(self.device)
        self.W_V = nn.Linear(self.config.d_hidden, self.config.n_head * self.config.d_head).to(self.device)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head*self.config.d_head, self.config.d_hidden).to(self.device)
        self.dropout = nn.Dropout(self.config.p_dropout).to(self.device)

    def forward(self, X_Q, X_K, X_V, attn_mask):
        # X_Q: (batch_size, n_q_seq, d_hidden), X_K: (batch_size, n_k_seq, d_hidden), X_V: (batch_size, n_v_seq, d_hidden), attn_mask: (batch_size, n_enc_seq, n_enc_seq)

        batch_size = X_Q.size(0)

        Q = self.W_Q(X_Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2) # Q: (batch_size, n_head, n_q_seq, d_head)
        K = self.W_K(X_K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2) # K: (batch_size, n_head, n_k_seq, d_head)
        V = self.W_V(X_V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2) # V: (batch_size, n_head, n_v_seq, d_head)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1) # attn_mask: (batch_size, n_head, n_q_seq, n_k_seq)

        context, attn_prob = self.scaled_dot_attn(Q, K, V, attn_mask)
        # context: (batch_size, n_head, n_q_seq, d_head), attn_prob: (batch_size, n_head, n_q_seq, n_k_seq)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head) # context: (batch_size, n_q_seq, n_head*d_head)

        output = self.dropout(self.linear(context)) # output: (batch_size, n_q_seq, d_hidden)

        return output, attn_prob

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidden, out_channels=self.config.d_ff, kernel_size=1).to(self.device)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidden, kernel_size=1).to(self.device)
        self.activation = F.relu
        self.dropout = nn.Dropout(self.config.p_dropout).to(self.device)

    def forward(self, inputs):
        # inputs: (batch_size, n_seq, d_hidden)

        output = self.activation(self.conv1(inputs.transpose(1,2))) # output: (batch_size, d_ff, n_seq)
        output = self.dropout(self.conv2(output).transpose(1,2)) # output: (batch_size, n_seq, d_hidden)

        return output

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device

        self.enc_self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidden, eps=self.config.layer_norm_epsilon).to(self.device)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidden, eps=self.config.layer_norm_epsilon).to(self.device)

    def forward(self, inputs, attn_mask):
        # inputs: (batch_size, n_enc_seq, d_hidden), attn_mask: (batch_size, n_enc_seq, n_enc_seq)

        attn_outputs, attn_prob = self.enc_self_attn(inputs, inputs, inputs, attn_mask)
        # attn_outputs: (batch_size, n_enc_seq, d_hidden), attn_prob: (batch_size, n_head, n_enc_seq, n_enc_seq)
        attn_outputs = self.layer_norm1(inputs + attn_outputs) # attn_outputs: (batch_size, n_enc_seq, d_hidden)

        ffn_outputs = self.pos_ffn(attn_outputs) # ffn_outputs: (batch_size, n_enc_seq, d_hidden)
        ffn_outputs = self.layer_norm2(ffn_outputs + attn_outputs) # attn_outputs: (batch_size, n_enc_seq, d_hidden)

        return ffn_outputs, attn_prob

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidden).to(self.device)
        sinusoid_table = torch.tensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_hidden), dtype=torch.float64).to(self.device)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True).to(self.device)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layers)])

    def forward(self, inputs):
        # inputs: (batch_size, n_enc_seq)

        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1 # positions: (batch_size, n_enc_seq)
        pos_mask = inputs.eq(self.config.i_padding)
        positions.masked_fill_(pos_mask, 0)

        outputs = self.enc_emb(inputs) + self.pos_emb(positions) # outputs: (batch_size, n_enc_seq, d_hidden)
        outputs = outputs.to(dtype=torch.float32)

        attn_mask = get_pad_mask(inputs, inputs, self.config.i_padding) # attn_mask: (batch_size, n_enc_seq, n_enc_seq)

        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            # outputs: (batch_size, n_enc_seq, d_hidden), attn_prob: (batch_size, n_head, n_enc_seq, n_enc_seq)
            attn_probs.append(attn_prob)

        return outputs, attn_probs

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device

        self.dec_self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidden, eps=self.config.layer_norm_epsilon).to(self.device)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidden, eps=self.config.layer_norm_epsilon).to(self.device)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidden, eps=self.config.layer_norm_epsilon).to(self.device)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_inputs: (batch_size, n_dec_seq, d_hidden), enc_outputs: (batch_size, n_enc_seq, d_hidden), dec_self_attn_mask: (batch_size, n_dec_seq, n_dec_seq), dec_enc_attn_mask: (batch_size, n_dec_seq, n_enc_seq)

        dec_self_attn_outputs, dec_self_attn_prob = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask) # dec_self_attn_outputs: (batch_size, n_dec_seq, d_hidden), dec_self_attn_prob: (batch_size, n_head, n_dec_seq, n_dec_seq)
        dec_self_attn_outputs = self.layer_norm1(dec_inputs + dec_self_attn_outputs)

        dec_enc_attn_outputs, dec_enc_attn_prob = self.dec_enc_attn(dec_self_attn_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask) # dec_enc_attn_outputs: (batch_size, n_dec_seq, d_hidden), dec_enc_attn_prob: (batch_size, n_head, n_dec_seq, n_enc_seq)
        dec_enc_attn_outputs = self.layer_norm2(dec_self_attn_outputs + dec_enc_attn_outputs)

        ffn_outputs = self.pos_ffn(dec_enc_attn_outputs) # ffn_outputs: (batch_size, n_dec_seq, d_hidden)
        ffn_outputs = self.layer_norm3(dec_enc_attn_outputs + ffn_outputs)

        return ffn_outputs, dec_self_attn_prob, dec_enc_attn_prob

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidden).to(self.device)
        sinusoid_table = torch.tensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_hidden), dtype=torch.float64).to(self.device)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True).to(self.device)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layers)])

        self.out = nn.Linear(self.config.d_hidden, self.config.n_dec_vocab).to(self.device)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # dec_inputs: (batch_size, n_dec_seq), enc_inputs: (batch_size, n_enc_seq), enc_outputs: (batch_size, n_enc_seq, d_hidden)

        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1 # (batch_size, n_dec_seq)
        pos_mask = dec_inputs.eq(self.config.i_padding)
        positions.masked_fill_(pos_mask, 0)

        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions) # dec_outputs: (batch_size, n_dec_seq, d_hidden)
        dec_outputs = dec_outputs.to(dtype=torch.float32)

        dec_attn_pad_mask = get_pad_mask(dec_inputs, dec_inputs, self.config.i_padding) # dec_attn_pad_mask: (batch_size, n_dec_seq, n_dec_seq)
        dec_attn_decoder_mask = get_subsequent_mask(dec_inputs) # dec_attn_decoder_mask: (batch_size, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0) # dec_self_attn_mask: (batch_size, n_dec_seq, n_dec_seq)
        dec_enc_attn_mask = get_pad_mask(dec_inputs, enc_inputs, self.config.i_padding) # dec_enc_attn_mask: (batch_size, n_dec_seq, n_enc_seq)

        dec_self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            # dec_outputs: (batch_size, n_dec_seq, d_hidden), dec_self_attn_prob: (batch_size, n_dec_seq, n_dec_seq), dec_enc_attn_prob: (batch_size, n_dec_seq, n_enc_seq)
            dec_self_attn_probs.append(dec_self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)

        dec_outputs = self.out(dec_outputs) # dec_outputs: (batch_size, n_dec_seq, n_dec_vocab)

        return dec_outputs, dec_self_attn_probs, dec_enc_attn_probs

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs: (batch_size, n_enc_seq), dec_inputs: (batch_size, n_dec_seq)

        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs) # enc_outputs: (batch_size, n_enc_seq, d_hidden)
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs) # dec_outputs: (batch_size, n_dec_seq, d_hidden)
#         dec_outputs = nn.Softmax(dim=-1)(dec_outputs)

        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
