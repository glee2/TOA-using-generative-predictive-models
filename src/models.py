# Notes
'''
Author: Gyumin Lee
Version: 1.1
Description (primary changes): Class to class
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/class2class/Tech_Gen/'
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
from transformers import DistilBertModel
from transformers import T5ForConditionalGeneration, T5Config

from utils import top_k_top_p_filtering

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

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch_size, n_enc_seq), attention_mask: (batch_size, n_enc_seq)

        inputs = input_ids

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

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config.device

        self.layers = self.set_layers(self.config.d_latent, self.config.d_pred_hidden, self.config.n_outputs, self.config.n_layers_predictor)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def set_layers(self, d_input, d_hidden, n_outputs, n_layers):
        layers = [
            nn.Sequential(
                nn.Linear(d_input, d_hidden),
                nn.ReLU()
            )
        ]
        for _ in range(n_layers-1):
            layers.append(
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    nn.ReLU()
                )
            )
        layers.append(nn.Sequential(nn.Linear(d_hidden, n_outputs)))
        layers = nn.ModuleList(layers)

        return layers

    def forward(self, x):
        # x (enc_outputs): (batch_size, n_enc_seq, d_latent)
        batch_size = x.size(0)
        #
        # if self.config.take_last_h:
        #     x = x[:, -1, :] # z: (batch_size, d_hidden) - Take last hidden states from enc_outputs
        # else:
        #     attn_weighted = self.attn_weight(x) # attn_weighted: (batch_size, n_enc_seq, 1)
        #     x = torch.bmm(attn_weighted.transpose(-1, 1), x).squeeze() # z: (batch_size, d_hidden)

        for layer in self.layers:
            x = layer(x)
        out = x if batch_size > 1 else x.unsqueeze(0)
        out = self.log_softmax(out)
        # out = self.sigmoid(out)
        # out = self.relu(out)

        return out

class Transformer(nn.Module):
    def __init__(self, config, tokenizers=None):
        super(Transformer, self).__init__()
        self.config = config
        self.device = self.config.device
        self.pooling_strategy = "max"
        if tokenizers is not None:
            self.tokenizers = tokenizers
        else:
            self.tokenizers = self.config.tokenizers

        if self.config.pretrained_enc:
            self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.encoder.config.output_attentions = True
        else:
            self.encoder = Encoder(self.config)
        self.predictor = Predictor(self.config) if "pred" in self.config.model_type else None
        if self.config.pretrained_dec:
            self.t5config = T5Config.from_pretrained("t5-small")
            self.t5model = T5ForConditionalGeneration.from_pretrained("t5-small", config=self.t5config).to(self.device)
            self.decoder = self.t5model.decoder
            self.lm_head = self.t5model.lm_head
            self.sizematching_layer = nn.Linear(self.config.d_enc_hidden, self.t5config.d_model)
            self.mu = nn.Linear(self.t5config.d_model, self.config.d_latent, bias=False)
            self.logvar = nn.Linear(self.t5config.d_model, self.config.d_latent, bias=False)
            self.embed_size_per_head = self.t5config.d_model // self.t5config.num_heads
            self.memory_projection = nn.Linear(
                self.config.d_latent,
                self.t5config.num_decoder_layers * self.t5config.num_heads * self.embed_size_per_head,
                bias=False,
            )
        else:
            self.decoder = Decoder(self.config) if "dec" in self.config.model_type else None
            self.t5model = self.lm_head = self.embed_size_per_head = self.memory_projection = None
            self.mu = nn.Linear(self.d_enc_hidden, self.config.d_latent, bias=False)
            self.logvar = nn.Linear(self.d_enc_hidden, self.config.d_latent, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs: (batch_size, n_enc_seq), dec_inputs: (batch_size, n_dec_seq)
        # if self.config.pretrained_enc:
        #     enc_outputs_base = self.encoder(**enc_inputs)
        #     enc_outputs, enc_self_attn_probs = enc_outputs_base.last_hidden_state, enc_outputs_base.attentions
        # else:
        #     enc_outputs, enc_self_attn_probs = self.encoder(**enc_inputs) # enc_outputs: (batch_size, n_enc_seq, d_hidden)
        #
        # pooled = self.pool(enc_outputs)
        # z, mu, logvar = self.calculate_latent(pooled)

        z = self.encode(enc_inputs)

        if self.predictor is not None:
            # pred_outputs = self.predictor(enc_outputs) # pred_outputs: (batch_size, n_outputs)
            # if self.config.take_last_h:
            #     z = enc_outputs[:, -1, :] # z: (batch_size, d_hidden) - Take last hidden states from enc_outputs
            # else:
            #     attn_weighted = self.predictor.attn_weight(enc_outputs) # attn_weighted: (batch_size, n_enc_seq, 1)
            #     z = torch.bmm(attn_weighted.transpose(-1, 1), enc_outputs).squeeze() # z: (batch_size, d_hidden)
            # pred_outputs = self.predictor(z) # pred_outputs: (batch_size, n_outputs)
            pred_outputs = self.predict(z)
        else:
            pred_outputs = None

        if self.decoder is not None:
        #     if self.config.pretrained_dec:
        #         past_key_values = self.build_past(z)
        #         dec_inputs = {"input_ids": dec_inputs["input_ids"][:,1:], "attention_mask": dec_inputs["attention_mask"]}
        #         dec_outputs = self.lm_head(self.decoder(**dec_inputs, past_key_values=past_key_values).last_hidden_state)
        #     else:
        #         dec_inputs = dec_inputs["input_ids"][:,:-1]
        #         dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs["input_ids"], enc_outputs) # dec_outputs: (batch_size, n_dec_seq, d_hidden)
        # else:
        #     dec_outputs = dec_self_attn_probs = dec_enc_attn_probs = None

        # return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs, enc_outputs, pred_outputs
            if self.config.pretrained_dec:
                dec_outputs = self.decode(dec_inputs=dec_inputs, z=z)
            else:
                dec_outputs = self.decode(dec_inputs=dec_inputs, enc_inputs=enc_inputs, enc_outputs=enc_outputs)

        return {"dec_outputs": dec_outputs, "z": z, "pred_outputs": pred_outputs}
        #, "enc_outputs": enc_outputs, "enc_self_attn_probs": enc_self_attn_probs, "dec_self_attn_probs": dec_self_attn_probs, "dec_enc_attn_probs": dec_enc_attn_probs}

    def freeze(self, module=None):
        if module=="decoder":
            for p in self.decoder.parameters():
                p.requires_grad = False
            if self.config.pretrained_dec:
                for p in self.lm_head.parameters():
                    p.requires_grad = False
                for p in self.memory_projection.parameters():
                    p.requires_grad = False
        elif module=="predictor":
            for p in self.predictor.parameters():
                p.requires_grad = False

    def defreeze(self, module=None):
        if module=="decoder":
            for p in self.decoder.parameters():
                p.requires_grad = True
            if self.config.pretrained_dec:
                for p in self.lm_head.parameters():
                    p.requires_grad = False
                for p in self.memory_projection.parameters():
                    p.requires_grad = False
            else:
                for p in self.decoder.pos_emb.parameters():
                    p.requires_grad = False
        elif module=="predictor":
            for p in self.predictor.parameters():
                p.requires_grad = True

    def encode(self, enc_inputs):
        if self.config.pretrained_enc:
            enc_outputs_base = self.encoder(**enc_inputs)
            enc_outputs, enc_self_attn_probs = enc_outputs_base.last_hidden_state, enc_outputs_base.attentions
        else:
            enc_outputs, enc_self_attn_probs = self.encoder(**enc_inputs) # enc_outputs: (batch_size, n_enc_seq, d_hidden)

        pooled = self.pool(enc_outputs)
        z, mu, logvar = self.calculate_latent(pooled)

        return z

    def predict(self, z):
        return self.predictor(z)

    def decode(self, enc_inputs=None, enc_outputs=None, dec_inputs=None, z=None):
        if self.config.pretrained_dec:
            batch_size = z.shape[0]
        else:
            batch_size = enc_inputs["input_ids"].shape[0]
        if dec_inputs is not None:
            if self.config.pretrained_dec:
                past_key_values = self.build_past(z)
                dec_inputs = {"input_ids": dec_inputs["input_ids"][:,1:], "attention_mask": dec_inputs["attention_mask"]}
                dec_outputs = self.lm_head(self.decoder(**dec_inputs, past_key_values=past_key_values).last_hidden_state)
            else:
                dec_inputs = dec_inputs["input_ids"][:,:-1]
                dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs["input_ids"], enc_outputs) # dec_outputs: (batch_size, n_dec_seq, d_hidden)
        else:
            if self.config.pretrained_dec:
                preds_recon_batch = torch.tile(torch.tensor(self.t5config.decoder_start_token_id, device=self.device), dims=(batch_size,1)).to(device=self.device)
                past_key_values = self.build_past(z)

                for i in range(self.config.n_dec_seq - 1):
                    dec_outputs = self.decoder(input_ids=preds_recon_batch[:, -1].unsqueeze(1), past_key_values=past_key_values)
                    logits = self.lm_head(dec_outputs.last_hidden_state)
                    past_key_values = dec_outputs.past_key_values
                    # print(logits.shape, past_key_values[0][0].shape)

                    # filtered_logits = top_k_top_p_filtering(logits, top_k=10, top_p=0.95)
                    # probabilites = F.softmax(filtered_logits, dim=-1)
                    # pred_tokens = torch.multinomial(probabilities, 1).unsqueeze(0)
                    pred_tokens = logits.argmax(2)[:,-1].unsqueeze(1)

                    preds_recon_batch = torch.cat([preds_recon_batch, pred_tokens], axis=1)
                    torch.cuda.empty_cache()

                dec_outputs = preds_recon_batch
            else:
                preds_recon_batch = torch.tile(torch.tensor(self.tokenizers["dec"].token_to_id("<SOS>"), device=self.device), dims=(batch_size,1)).to(device=self.device)

                for i in range(self.config.n_dec_seq - 1):
                    dec_outputs, *_ = self.decoder(preds_recon_batch, enc_inputs["input_ids"], enc_outputs)
                    pred_tokens = dec_outputs.argmax(2)[:,-1].unsqueeze(1)
                    preds_recon_batch = torch.cat([preds_recon_batch, pred_tokens], axis=1)
                    torch.cuda.empty_cache()

                dec_outputs = preds_recon_batch

        return dec_outputs

    def pool(self, x):
        # Shape of x - (layer_count, batch_size, seq_length, hidden_size)
        # x = torch.stack(x[1:])
        # x = x.transpose(0, 1)
        if self.pooling_strategy == "mean":
            # return x[:, -1, :, :].mean(dim=1)
            out = x.mean(dim=1)
        elif self.pooling_strategy == "max":
            # return torch.max(x[:, -1, :, :], dim=1)[0]  # Pool from last layer.
            out = torch.max(x, dim=1)[0]
        else:
            raise Exception("Wrong pooling strategy!")

        if self.config.pretrained_dec:
            if x.shape[1] != self.t5config.d_model:
                out = self.sizematching_layer(out)

        return out

    def calculate_latent(self, pooled):
        mu, logvar = self.mu(pooled), self.logvar(pooled)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def build_past(self, z):
        projection = self.memory_projection(z)
        cross_attn = projection.reshape(
            self.t5config.num_decoder_layers,
            projection.shape[0],
            self.t5config.num_heads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values

# ========================================== #
# PRETRAINED model for decoder (current: T5) #
# ========================================== #

class ModifiedT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config, latent_dim, pooling_strategy):
        super().__init__(config)
        self.latent_dim = latent_dim
        self.embed_size_per_head = config.d_model // config.num_heads
        self.memory_projection = nn.Linear(
            latent_dim,
            config.num_decoder_layers * config.num_heads * self.embed_size_per_head,
            bias=False,
        )
        self.pooling_strategy = pooling_strategy

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sampled_z=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        z, mu, logvar = None, None, None
        if sampled_z is not None:
            z = sampled_z
            encoder_outputs = BaseModelOutput(
                last_hidden_state=None,
                hidden_states=None,
                attentions=None,
            )
        elif encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.run_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled = self.pool(encoder_outputs.hidden_states)
            z, mu, logvar = self.calculate_latent(pooled)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if past_key_values is None:
            past_key_values = self.build_past(z)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None and labels is None:
            # assert (
            #    labels is None
            # ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            # hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            pass

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        out = Seq2SeqLMOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        out.mu = mu
        out.logvar = logvar
        out.z = z
        return out

    def run_encoder(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True

        return encoder_outputs

    def build_past(self, z):
        projection = self.memory_projection(z)
        cross_attn = projection.reshape(
            self.config.num_decoder_layers,
            projection.shape[0],
            self.config.num_heads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values
