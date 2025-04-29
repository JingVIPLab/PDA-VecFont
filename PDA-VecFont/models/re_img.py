from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from utils.resnet import resnet34
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


class decoder_block(nn.Module):
    def __init__(self, channels, kernel_size, stride, output_padding, img_size):
        super(decoder_block, self).__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(1, channels, kernel_size, stride, padding)
        self.LN1 = nn.LayerNorm((channels, img_size, img_size))
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)

        self.deconv = nn.ConvTranspose2d(channels, 1, kernel_size, stride, padding, output_padding)
        self.LN2 = nn.LayerNorm((channels, img_size, img_size))

        self.conv2 = nn.Conv2d(2 , 1, 3, 1, 1)
        self.LN2 = nn.LayerNorm((1, img_size*2, img_size*2))


        self.seq_k = nn.Linear(512, channels//4)
        self.seq_v = nn.Linear(512, channels)

        self.image_q1 = nn.Linear(channels, channels//4)
        self.image_k1 = nn.Linear(channels, channels)
        self.attn_LN1 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(p=0.1)


    def forward(self, x, seq_feat ):


        seq_v = self.seq_v(seq_feat)
        seq_k = self.seq_k(seq_feat)
        x_c = self.conv1(x)
        x_c = self.LN1(x_c)
        x_c = self.ReLU(x_c)

        B, C, H, W = x_c.shape
        x1 = x_c.view(B, C, H * W).transpose(-2, -1)
        image_q1 = self.image_q1(x1)
        image_k1 = self.image_k1(x1)
        attn_x1 = self.attn_LN1(image_attention(image_q1, seq_k, seq_v, dropout=self.drop) + x1)
        x1 = image_attention(attn_x1, image_k1, x1, dropout=self.drop)
        x1 = x1.transpose(-2, -1).view(B, C, H, W)

        x1 = self.deconv(x1)
        x1 = self.LN2(x1)
        x1 = self.ReLU(x1)

        x_merge = torch.cat([x, x1], dim=1)
        x_merge = self.conv2(x_merge)
        return torch.sigmoid(x_merge)

# if __name__ == '__main__':
#
#      decoder = decoder_block(16, 3, 2, 1, 32).cuda()
#      ref_img = torch.tensor(np.ones(( 16, 1, 64, 64))).float().cuda()
#      seq_feat = torch.tensor(np.ones((16, 51,  512))).float().cuda()
#      img_feat = torch.tensor(np.ones((16, 512))).float().cuda()
#      x = decoder(ref_img,seq_feat)
#      print(x)