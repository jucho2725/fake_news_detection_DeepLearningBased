import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class Multihead(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(Multihead, self).__init__()
        self.d_k = d_model // head
        self.head = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn= None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        n_batch = query.size(0)
        query, key, value = [l(x).view(n_batch, -1, self.head, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linears[-1](x)


class Pff(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Pff, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        return self.w_2(self.dropout((x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerCon(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerCon, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerCon(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Classifier_cls_ent(nn.Module):
    def __init__(self, full_encoder_cls, full_encoder_ent, dropout=0.1):
        super(Classifier_cls_ent, self).__init__()
        self.fec = full_encoder_cls
        self.few = full_encoder_ent

        self.linear_1 = nn.Linear(4608, 1024)
        self.linear_2 = nn.Linear(1024, 128)
        self.linear_3 = nn.Linear(128, 16)
        self.linear_4 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(self, x, y):
        x = self.fec(x)
        x_1 = torch.cat((x[:, 0, :], x[:, -1, :]), dim=1)
        y = self.few(y)
        y_1 = torch.cat((y[:, 0, :], y[:, 63, :], y[:, 64, :], y[:, -1, :]), dim=1)
        z = torch.cat((x_1, y_1), dim=1)

        z = self.linear_1(z)
        z = self.dropout_1(self.linear_2((z * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))))))
        z = self.dropout_2(self.linear_3((z * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))))))
        z = self.dropout_3(self.linear_4((z * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))))))
        return self.sigmoid(z)


def init_model(N=9, d_model=768, d_ff=3072, head=12, dropout=0.1):
    c = copy.deepcopy
    attn = Multihead(head, d_model, dropout)
    ff = Pff(d_model, d_ff, dropout)
    model = Classifier_cls_ent(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                               Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model