import torch
import torch.nn as nn
import torchvision.models as models
from models_SDT.transformer import *
from einops import rearrange
import numpy as np
from models_SDT.loss import SupConLoss


class Encoder_part(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_head_layers=1,
                 dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=True):
        super(Encoder_part, self).__init__()
        ### style encoder with dual heads
        self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(
            models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        glyph_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.glyph_head = TransformerEncoder(encoder_layer, num_head_layers, glyph_norm)

        ### two mlps that project style features into the space where nce_loss is applied

        self.pro_mlp_character = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))

        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        self._reset_parameters()
        self.nce_loss = SupConLoss(contrast_mode='all')

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def random_double_sampling(self, x, ratio=0.25):
        """
        Sample the positive pair (i.e., o and o^+) within a character by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [L, B, N, D], sequence
        return o [B, N, 1, D], o^+ [B, N, 1, D]
        """
        L, B, N, D = x.shape  # length, batch, group_number, dim
        x = rearrange(x, "L B N D -> B N L D")
        noise = torch.rand(B, N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)

        anchor_tokens, pos_tokens = int(L * ratio), int(L * 2 * ratio)
        ids_keep_anchor, ids_keep_pos = ids_shuffle[:, :, :anchor_tokens], ids_shuffle[:, :, anchor_tokens:pos_tokens]
        x_anchor = torch.gather(
            x, dim=2, index=ids_keep_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
        x_pos = torch.gather(
            x, dim=2, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 1, D))
        return x_anchor, x_pos

    # the shape of style_imgs is [B, 2*N, C, H, W] during training
    def forward(self, style_imgs):
        style_imgs = torch.unsqueeze(style_imgs,dim=2)
        batch_size, anchor_num, in_planes, h, w = style_imgs.shape

        # style_imgs: [B, 2*N, C:1, H, W] -> FEAT_ST_ENC: [4*N, B, C:512]
        style_imgs = style_imgs.view(-1, in_planes, h, w)  # [B*2N, C:1, H, W]
        style_embe = self.Feat_Encoder(style_imgs)  # [B*2N, C:512, 2, 2]

        style_embe = style_embe.view(batch_size * anchor_num, 512, -1).permute(2, 0, 1)  # [4, B*2N, C:512]
        FEAT_ST_ENC = self.add_position(style_embe)

        memory = self.base_encoder(FEAT_ST_ENC)  # [4, B*N, C]

        glyph_memory = self.glyph_head(memory)


        glyph_memory = rearrange(glyph_memory, 't (b  n) c -> t  b  n  c',
                                 b=batch_size,  n=anchor_num)  # [4, B, N, C]

        # glyph-nce
        # sample the positive pair
        anc, positive = self.random_double_sampling(glyph_memory)
        n_channels = anc.shape[-1]
        anc = anc.reshape(batch_size, -1, n_channels)
        anc_compact = torch.mean(anc, 1, keepdim=True)
        anc_compact = self.pro_mlp_character(anc_compact)  # [B, 1, C]
        positive = positive.reshape(batch_size, -1, n_channels)
        positive_compact = torch.mean(positive, 1, keepdim=True)
        positive_compact = self.pro_mlp_character(positive_compact)  # [B, 1, C]

        nce_emb_patch = torch.cat((anc_compact, positive_compact), 1)  # [B, 2, C]
        nce_emb_patch = nn.functional.normalize(nce_emb_patch, p=2, dim=2)
        nce_loss_glyph = self.nce_loss(nce_emb_patch)

        # input the writer-wise & character-wise styles into the decoder
        glyph_style = glyph_memory # [4, B, N, C]
        #L, B, N, D = glyph_style.shape  # length, batch, group_number, dim
        glyph_style = rearrange(glyph_style, "L B N D -> B N (L D)")
        img_feat = torch.mean(glyph_style,dim=1)


        return  img_feat, nce_loss_glyph

# if __name__ == '__main__':
#     imgfeat = torch.tensor(np.ones((16, 8, 64, 64))).float().cuda()
#
#     encoder = Encoder_part().cuda()
#
#     output , emb = encoder(imgfeat)
#     print(output.shape)
#     print(emb)