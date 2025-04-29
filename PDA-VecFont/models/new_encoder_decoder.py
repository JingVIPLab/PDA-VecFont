from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class encoder_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, img_size):
        super(encoder_block, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.LN1 = nn.LayerNorm((output_channels, img_size, img_size))
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)
        #self.conv2 = nn.Conv2d(input_channels, output_channels, 1, stride, 0)
        #self.LN2 = nn.LayerNorm((output_channels, img_size, img_size))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.LN1(x1)
        x1 = self.ReLU(x1)
        #x2 = self.conv2(x)
        #x2 = self.LN2(x2)
        return x1


class encoder(nn.Module):
    def __init__(self, input_channels, img_size, ngf):
        super(encoder, self).__init__()
        channels = [input_channels, ngf, 2*ngf, 4*ngf, 8*ngf, 16*ngf, 32*ngf, 64*ngf ]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3 ]
        strides = [1, 2, 2, 2, 2, 2, 2]
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            img_size = (img_size + kernel_sizes[i] // 2 * 2 - (kernel_sizes[i]-1) - 1) // strides[i] + 1
            block = encoder_block(channels[i], channels[i+1], kernel_sizes[i], strides[i], img_size)
            self.blocks.append(block)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)
            

        return x


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
    def __init__(self, input_channels, output_channels, kernel_size, stride, output_padding, img_size, attn):
        super(decoder_block, self).__init__()
        padding = kernel_size // 2
        self.deconv1 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding)
        self.LN1 = nn.LayerNorm((output_channels, img_size, img_size))
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)
        #self.deconv2 = nn.ConvTranspose2d(input_channels, output_channels, 1, stride, 0, output_padding)
        #self.LN2 = nn.LayerNorm((output_channels, img_size, img_size))
        self.attn = attn
        if attn:
            self.seq_k = nn.Linear(512, output_channels // 4)
            self.seq_v = nn.Linear(512, output_channels)
            self.image_q1 = nn.Linear(output_channels, output_channels // 4)
            #self.image_q2 = nn.Linear(output_channels, output_channels // 4)
            self.image_k1 = nn.Linear(output_channels, output_channels)
           # self.image_k2 = nn.Linear(output_channels, output_channels)
            self.attn_LN1 = nn.LayerNorm(output_channels)
           # self.attn_LN2 = nn.LayerNorm(output_channels)
            self.drop = nn.Dropout(p=0.1)

    def forward(self, x, seq_feat):
        x1 = self.deconv1(x)
        x1 = self.LN1(x1)
        x1 = self.ReLU(x1)

        #x2 = self.deconv2(x)
        #x2 = self.LN2(x2)
        if self.attn:
            seq_v = self.seq_v(seq_feat)
            seq_k = self.seq_k(seq_feat)

            B, C, H, W = x1.shape
            x1 = x1.view(B, C, H * W).transpose(-2, -1)
            image_q1 = self.image_q1(x1)
            image_k1 = self.image_k1(x1)
            attn_x1 = self.attn_LN1(image_attention(image_q1, seq_k, seq_v, dropout=self.drop) + x1)
            x1 = image_attention(attn_x1, image_k1, x1, dropout=self.drop)
            x1 = x1.transpose(-2, -1).view(B, C, H, W)

            # B, C, H, W = x2.shape
            # x2 = x2.view(B, C, H * W).transpose(-2, -1)
            # image_q2 = self.image_q2(x2)
            # image_k2 = self.image_k2(x2)
            # attn_x2 = self.attn_LN2(image_attention(image_q2, seq_k, seq_v, dropout=self.drop) + x2)
            # x2 = image_attention(attn_x2, image_k2, x2, dropout=self.drop)
            # x2 = x2.transpose(-2, -1).view(B, C, H, W)

            x1 = self.LN1(x1)
            x1 = self.ReLU(x1)
            # x2 = self.LN2(x2)
            # x2 = self.ReLU(x2)
        return x1
#修改输入维度与尺寸，不需要两个卷积相加，直接就可以。

class decoder(nn.Module):

    def __init__(self, img_size,  ndf):
        super(decoder, self).__init__()
        channels = [32 * ndf, 16 * ndf, 8 * ndf, 4 * ndf, 2 * ndf, ndf]
        kernel_sizes = [3, 3, 5, 5, 5]
        strides = [2, 2, 2, 2, 2]
        output_paddings = [ 1, 1, 1, 1, 1]
        attn = [0, 0, 0, 0, 0]
        self.catchar = nn.Sequential(nn.ConvTranspose2d(564, 512,kernel_size=3, stride=2,padding=3 // 2, output_padding=1),
                       nn.LayerNorm([512, 2, 2]),
                       nn.ReLU(True))


        self.blocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            img_size = (img_size - 1) * strides[i] - kernel_sizes[i] // 2 * 2 + kernel_sizes[i] + output_paddings[i]
            block = decoder_block( channels[i], channels[i + 1], kernel_sizes[i], strides[i], output_paddings[i],
                                  img_size, attn[i])
            self.blocks.append(block)

        self.torgb2 = nn.Conv2d(ndf , 1, 3, 1, 1)  # 输出卷积

    def forward(self, imgfeat , char_onehot, seq_feat):
        x = torch.cat((imgfeat, char_onehot), dim=-1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.catchar(x)
        for block in self.blocks:
            x = block(x, seq_feat)
        x = self.torgb2(x)
        return torch.sigmoid(x)

# if __name__ == '__main__':
#
#
#      ref_img=torch.tensor(np.ones((16, 4, 64, 64))).float().cuda()
#      content_encoder = encoder(4, 64, 64).cuda()
#      seq_feat = torch.tensor(np.ones((16, 512))).float().cuda()
#      char_onehot = torch.tensor(np.ones((16, 52))).float().cuda()
#      x,output = content_encoder(ref_img)
#      x = x.view(x.shape[0],x.shape[1])
#      print(x.shape)
#      #summary(content_encoder, input_size=(4, 64, 64), batch_size=16, device="cuda")
#
#      decoder=decoder(2,16).cuda()
#      print('decoder')
#      out_img = decoder(x,char_onehot,seq_feat)
#
#      print(out_img.shape)

    # #summary(decoder, input_size=[(512, 1, 1),(512,)], batch_size=16, device="cuda")
    #  model = image_branch().cuda()

     #out_img = model(ref_img,char_onehot,seq_feat) # 融合模块
    # print(out_img['fake'].shape,out_img['temp'])