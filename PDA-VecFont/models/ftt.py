from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.distributions as DIS
import numpy as np
import yaml







def image_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value)


class FTT(nn.Module):
    def __init__(self, channels,  img_size):
        super(FTT, self).__init__()

        self.LN2 = nn.LayerNorm((img_size*img_size,channels))
        self.seq_k = nn.Linear(512, channels//4)
        self.seq_v = nn.Linear(512, channels)

        self.image_q1 = nn.Linear(channels, channels//4)
        self.image_k1 = nn.Linear(channels, channels)
        self.attn_LN1 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(p=0.1)


    def forward(self, x, seq_feat):


        seq_v = self.seq_v(seq_feat)
        seq_k = self.seq_k(seq_feat)

        B, C, H, W = x.shape
        x1 = x.view(B, C, H * W).transpose(-2, -1)
        image_q1 = self.image_q1(x1)
        image_k1 = self.image_k1(x1)

        attn_x1 = self.attn_LN1(image_attention(image_q1, seq_k, seq_v, dropout=self.drop) + x1)
        x1 = image_attention(attn_x1, image_k1, x1, dropout=self.drop)
        x1 = self.LN2(x1)


        return x1