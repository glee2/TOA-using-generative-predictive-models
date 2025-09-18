# Notes
'''
Author: Gyumin Lee
Version: 2.0
Description (primary changes): Code refactoring
'''

# Set root directory
root_dir = '/home2/glee/dissertation/1_tech_gen_impact/class2class/Tech_Gen/'
import sys
sys.path.append(root_dir)

# Basic libraries
import pandas as pd
import numpy as np

# DL libraries
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertModel, T5Config, T5ForConditionalGeneration

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
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.tokenizer = tokenizer

        self.dec_emb = nn.Embedding(self.tokenizer.vocab_size, self.config.d_hidden).to(self.device)
        sinusoid_table = torch.tensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_hidden), dtype=torch.float64).to(self.device)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True).to(self.device)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layers)])

        self.out = nn.Linear(self.config.d_hidden, self.tokenizer.vocab_size).to(self.device)

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

class VCLS2CLS(nn.Module):
    def __init__(self, config={}, tokenizers=None):
        super(VCLS2CLS, self).__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        self.tokenizers = tokenizers
        self.hidden_factor = self.n_directions * self.n_layers

        if self.pretrained_enc:
            claim_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
            claim_encoder.config.output_attentions = True
            claim_encoder.d_hidden = 768
            for p in claim_encoder.parameters():
                p.requires_grad = False
        else:
            claim_encoder = Encoder_Transformer(self.config, tokenizer=self.tokenizers["claim_enc"])

        class_encoder = Encoder_SEQ(config=self.config, tokenizer=self.tokenizers["class_enc"])
        self.encoders = nn.ModuleDict({"claim": claim_encoder, "class": class_encoder})

        self.latent2hidden = nn.Linear(self.d_latent, self.d_hidden * self.hidden_factor)
        self.decoder = Decoder_SEQ(config=self.config, tokenizer=self.tokenizers["class_dec"])
        self.predictor = Predictor(self.config) if "pred" in self.model_type else None
        self.pooling_strategy = "max"
        self.mu = nn.Linear(self.d_hidden + (self.d_hidden * self.n_directions * self.n_layers), self.d_latent, bias=False)
        self.logvar = nn.Linear(self.d_hidden + (self.d_hidden * self.n_directions * self.n_layers), self.d_latent, bias=False)

    def forward(self, enc_inputs, dec_inputs, teach_force_ratio=0.75):
        if isinstance(enc_inputs, dict):
            batch_size = enc_inputs["class"].size(0)
        else:
            batch_size = enc_inputs.size(0)

        enc_outputs, z, mu, logvar = self.encode(enc_inputs)
        pred_outputs = self.predict(z)
        if z.device != self.device:
            curr_device = z.device
        else:
            curr_device = self.device
        dec_outputs = self.decode(z, enc_outputs["class"], dec_inputs, device=curr_device, teach_force_ratio=teach_force_ratio)

        return {"dec_outputs": dec_outputs, "z": z, "pred_outputs": pred_outputs, "mu": mu, "logvar": logvar}

    def freeze(self, module_name, defreeze=False):
        modules = {"decoder": self.decoder, "predictor": self.predictor, "claim_encoder": self.encoders["claim"], "class_encoder": self.encoders["class"]}
        module = modules[module_name]
        for p in module.parameters():
            p.requires_grad = defreeze

    def encode(self, enc_inputs):
        batch_size = enc_inputs["class"].size(0)
        enc_outputs = {}
        hiddens = {}
        if self.pretrained_enc:
            enc_outputs_base = self.encoders["claim"](**enc_inputs["claim"])
            enc_outputs["claim"] = enc_outputs_base.last_hidden_state
        else:
            enc_outputs["claim"], *_ = self.encoders["claim"](**enc_inputs["claim"]) # enc_outputs: (batch_size, n_enc_seq, d_hidden)

        hiddens["claim"] = self.pool(enc_outputs["claim"])

        enc_outputs["class"], hiddens["class"] = self.encoders["class"](enc_inputs["class"]) # enc_outputs: (batch_size, n_enc_seq_class, d_hidden * n_directions), hidden: (hidden_factor, batch_size, d_hidden)

        if self.bidirec or self.n_layers > 1:
            # flatten hidden states
            hiddens["class"] = hiddens["class"].permute(1,0,2).contiguous().view(batch_size, -1)
        else:
            hiddens["class"] = hiddens["class"].squeeze()

        hiddens_cat = torch.cat(list(hiddens.values()), dim=1)
        z, mu, logvar = self.calculate_latent(hiddens_cat)

        return enc_outputs, z, mu, logvar

    def predict(self, z):
        return self.predictor(z)

    def decode(self, z, enc_outputs, dec_inputs=None, batch_size=None, device=None, teach_force_ratio=0.75):
        if device is None:
            device = self.device
        if batch_size is None:
            batch_size = z.size(0)

        next_hidden = self.latent2hidden(z)

        if self.bidirec or self.n_layers > 1:
            # unflatten hidden states
            next_hidden = next_hidden.view(batch_size, self.d_hidden, self.hidden_factor).permute(2,0,1).contiguous()
        else:
            next_hidden = next_hidden.unsqueeze(0)

        next_input = torch.tensor(np.tile([self.tokenizers["class_enc"].vocab_w2i["<SOS>"]], batch_size), device=device) # (batch_size)

        dec_outputs = torch.zeros((batch_size, self.n_dec_seq_class, self.tokenizers["class_dec"].get_vocab_size()), device=device) # (batch_size, vocab_size, seq_len)

        for t in range(1, self.n_dec_seq_class):
            output, next_hidden, pred_token = self.pred_next(next_input, next_hidden, enc_outputs)
            # output: (batch_size, vocab_size), hidden: (batch_size, hidden_dim * n_directions), pred_token: (batch_size)

            if dec_inputs is not None:
                rand_num = np.random.random()
                if rand_num < teach_force_ratio:
                    next_input = dec_inputs[:,t] # (batch_size)
                else:
                    next_input = pred_token # (batch_size)
            else:
                next_input = pred_token
            dec_outputs[:,t,:] = output

        return dec_outputs

    def pred_next(self, next_input, next_hidden, enc_outputs):
        # output, hidden = self.decoder(next_input, next_hidden, enc_outputs)
        output, hidden = self.decoder(next_input, next_hidden)
        # output: (batch_size, vocab_size), hidden: (n_layers, batch_size, hidden_dim * n_directions)

        pred_token = output.argmax(1) # (batch_size)
        if output.size(0) != 1: pred_token = pred_token.squeeze(0)

        if pred_token.size() == torch.Size([]):
            print(output.shape, hidden.shape, pred_token.shape, next_input.shape)

        return output, hidden, pred_token

    def pool(self, x):
        if self.pooling_strategy == "mean":
            out = x.mean(dim=1)
        elif self.pooling_strategy == "max":
            out = torch.max(x, dim=1)[0]
        else:
            raise Exception("Wrong pooling strategy!")

        return out

    def calculate_latent(self, pooled):
        mu, logvar = self.mu(pooled), self.logvar(pooled)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        # self.device = self.config.device

        self.layers = self.set_layers(self.d_latent, self.d_pred_hidden, self.n_outputs, self.n_layers_predictor)
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
        batch_size = x.size(0)
        for layer in self.layers:
            x = layer(x)
        # out = x if batch_size > 1 else x.unsqueeze(0)
        out = x
        out = self.log_softmax(out)

        return out

class Encoder_Transformer(nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        # self.device = self.config.device
        self.tokenizer = tokenizer
        # self.d_hidden =  self.d_hidden

        self.enc_emb = nn.Embedding(self.tokenizer.get_vocab_size(), self.d_hidden)
        sinusoid_table = torch.tensor(get_sinusoid_encoding_table(self.n_enc_seq_claim + 1, self.d_hidden), dtype=torch.float32)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.n_layers)])

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch_size, n_enc_seq), attention_mask: (batch_size, n_enc_seq)

        inputs = input_ids

        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1 # positions: (batch_size, n_enc_seq)
        pos_mask = inputs.eq(self.i_padding)
        positions.masked_fill_(pos_mask, 0)

        outputs = self.enc_emb(inputs) + self.pos_emb(positions) # outputs: (batch_size, n_enc_seq, d_hidden)
        outputs = outputs.to(dtype=torch.float32)

        attn_mask = get_pad_mask(inputs, inputs, self.i_padding) # attn_mask: (batch_size, n_enc_seq, n_enc_seq)

        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            # outputs: (batch_size, n_enc_seq, d_hidden), attn_prob: (batch_size, n_head, n_enc_seq, n_enc_seq)
            attn_probs.append(attn_prob)

        return outputs, attn_probs

class Encoder_SEQ(nn.Module):
    def __init__(self, config={}, tokenizer=None):
        super(Encoder_SEQ, self).__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        # self.device = self.config.device
        self.tokenizer = tokenizer
        # self.d_hidden =  self.config.d_enc_hidden * self.config.n_directions

        self.gru = nn.GRU(self.d_embedding, self.d_hidden, self.n_layers, batch_first=True, bidirectional=self.bidirec).to(self.device)
        self.embedding = nn.Embedding(self.tokenizer.get_vocab_size(), self.d_embedding, padding_idx=self.i_padding).to(self.device)
        self.dropout = nn.Dropout(self.p_dropout).to(self.device)
        # self.fc = nn.Linear(self.d_hidden, self.config.d_enc_hidden).to(self.device)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len)
        batch_size = inputs.shape[0]

        embedded = self.dropout(self.embedding(inputs)) # (batch_size, seq_len, embedding_dim)
        hidden = self.initHidden(batch_size) # (n_layers * n_directions, batch_size, hidden_dim)
        if inputs.device != hidden.device: hidden = hidden.to(inputs.device)
        batch_lengths = (inputs==self.tokenizer.eos_id).nonzero()[:,-1]
        sorted_lengths, sorted_idxs = torch.sort(batch_lengths, descending=True)
        packed_input = pack_padded_sequence(embedded, sorted_lengths.data.tolist(), batch_first=True)
        output, hidden = self.gru(packed_input, hidden)
        padded_output, *_ = pad_packed_sequence(output, total_length=self.n_enc_seq_class, batch_first=True)
        padded_output = padded_output.contiguous()
        _, reversed_idxs = torch.sort(sorted_idxs)
        padded_output = padded_output[reversed_idxs]
        output = padded_output

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros((self.n_layers * self.n_directions, batch_size, self.d_hidden), device=self.device)

class Decoder_SEQ(nn.Module):
    def __init__(self, config={}, tokenizer=None):
        super().__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.embedding = nn.Embedding(self.tokenizer.get_vocab_size(), self.d_embedding)
        self.dropout = nn.Dropout(self.p_dropout)
        self.gru = nn.GRU(self.d_embedding, self.d_hidden, self.n_layers, batch_first=True, bidirectional=self.bidirec)
        self.fc_out = nn.Linear(self.d_hidden * self.n_directions, self.tokenizer.get_vocab_size())

    def forward(self, inputs, hidden=None):
        inputs = inputs.unsqueeze(1)

        if hidden is not None:
            if len(hidden.size()) < 3: # last hidden state from encoder (when starting decoding)
                hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        else:
            hidden = self.initHidden(len(inputs))
            if inputs.device != hidden.device: hidden = hidden.to(inputs.device)

        embedded = self.dropout(self.embedding(inputs))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        # prediction = self.log_softmax(prediction)

        return prediction, hidden

    def initHidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.d_hidden))

class Attention(nn.Module):
    def __init__(self, config={}):
        super(Attention, self).__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        self.d_attn_hidden = self.d_hidden + (self.d_hidden * self.n_directions)
        self.attn = nn.Linear(self.d_attn_hidden, self.d_hidden).to(self.device)
        self.v = nn.Linear(self.d_hidden, 1, bias=False).to(self.device)

    def forward(self, hidden, enc_outputs):
        # hidden: (n_layers, batch_size, hidden_dim * n_directions), encoder_outputs: (batch_size, seq_len, hidden_dim * n_directions)

        batch_size = enc_outputs.shape[0]
        seq_len = enc_outputs.shape[1]

        hidden_last = hidden[-1] # (batch_size, hidden_dim * n_directions)

        hidden = hidden_last.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_dim * n_directions)

        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2))) # (batch_size, seq_len, hidden_dim)
        attention = self.v(energy).squeeze(2) # (batch_size, seq_len)

        return F.softmax(attention, dim=1)

class AttnDecoder_SEQ(nn.Module):
    def __init__(self, config={}, tokenizer=None):
        super().__init__()
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.attention = Attention(config=self.config)
        self.embedding = nn.Embedding(self.tokenizer.get_vocab_size(), self.d_embedding).to(self.device)
        self.gru = nn.GRU((self.d_hidden * self.n_directions) + self.d_embedding, self.d_hidden, self.n_layers, bidirectional=self.bidirec, batch_first=True).to(self.device)
        self.fc_out = nn.Linear(self.d_embedding + (self.d_hidden * self.n_directions) + (self.d_hidden * self.n_directions), self.tokenizer.get_vocab_size())
        self.dropout = nn.Dropout(self.p_dropout).to(self.device)

    def forward(self, inputs, hidden, enc_outputs):
        # inputs: (batch_size), hidden: (n_layers, batch_size, hidden_dim * n_directions), encoder_outputs: (batch_size, seq_len, hidden_dim * n_directions)
        inputs = inputs.unsqueeze(1) # (batch_size, 1)

        embedded = self.dropout(self.embedding(inputs)) # (batch_size, 1, embedding_dim)

        a = self.attention(hidden, enc_outputs)
        a = a.unsqueeze(1) # (batch_size, 1, seq_len)

        weighted = torch.bmm(a, enc_outputs) # (batch_size, 1, hidden_dim * n_directions)

        gru_input = torch.cat((embedded, weighted), dim=2) # (batch_size, 1, hidden_dim * n_directions + embedding_dim)

        output, hidden = self.gru(gru_input, hidden) # output: (batch_size, 1, hidden_dim * n_directions), hidden: (n_layers, batch_size, hidden_dim * n_directions)

        embedded = embedded.squeeze(1)
        weighted = weighted.squeeze(1)
        output = output.squeeze(1)

        prediction = self.fc_out(torch.cat((embedded, weighted, output), dim=1)) # (batch_size, vocab_size)
        # prediction = self.log_softmax(prediction)

        return prediction, hidden
