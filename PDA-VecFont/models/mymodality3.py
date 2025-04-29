import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from options import get_parser_main_model

opts = get_parser_main_model().parse_args()


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class ModalityFusion(nn.Module):
    def __init__(self, img_size=64, ref_nshot=4, bottleneck_bits=512, ngf=32, seq_latent_dim=512, mode='train',skip=False):
        super().__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.ref_nshot = ref_nshot
        self.mode = mode
        self.fc_merge = nn.Linear(seq_latent_dim * opts.ref_nshot, 512)
        n_downsampling = int(math.log(img_size, 2))
        mult_max = 2 ** (n_downsampling)
        self.fc_fusion = nn.Linear(opts.bottleneck_bits*4 , opts.bottleneck_bits * 2,
                                   bias=True)# the max multiplier for img feat channels is
        self.fc_seq = nn.Linear(opts.bottleneck_bits , opts.bottleneck_bits * 2,
                                   bias=True)
        self.fc_mergeimgseq = nn.Linear(opts.bottleneck_bits * 2, opts.bottleneck_bits)
        self.fc_forward_img = nn.Linear(opts.bottleneck_bits , opts.bottleneck_bits)
        self.fc_forward_seq = nn.Linear(opts.bottleneck_bits , opts.bottleneck_bits)
        

        self.img_mod = nn.Linear(512, 256, bias=False)  # 特征压缩
        self.img_demod = nn.Linear(256, 512, bias=False)  # 特征升维
        self.seq_mod = nn.Linear(512, 256, bias=False)
        self.seq_demod = nn.Linear(256, 512, bias=False)

        self.celoss = nn.CrossEntropyLoss()
        self.l2loss = nn.MSELoss()
        self.skip = skip
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, seq_feat, img_feat, ref_pad_mask=None):

        # 处理sqe特征
        cls_one_pad = torch.ones((1, 1, 1)).to(seq_feat.device).repeat(seq_feat.size(0), 1, 1)
        ref_pad_mask = torch.cat([cls_one_pad, ref_pad_mask], dim=-1)

        seq_feat = seq_feat * (ref_pad_mask.transpose(1, 2))
        seq_feat_ = seq_feat.view(seq_feat.size(0) // self.ref_nshot, self.ref_nshot, seq_feat.size(-2),
                                  seq_feat.size(-1))
        seq_feat_ = seq_feat_.transpose(1, 2)
        seq_feat_ = seq_feat_.contiguous().view(seq_feat_.size(0), seq_feat_.size(1),
                                                seq_feat_.size(2) * seq_feat_.size(3))
        seq_feat_ = self.fc_merge(seq_feat_)
        seq_feat_cls = seq_feat_[:, 0]
        seq_feat_r = seq_feat_cls.clone()
        #seq_feat_mod = F.normalize(self.seq_mod(seq_feat_r), dim=1)
        #seq_feat_demod = self.seq_demod(seq_feat_mod)
        #epsilon = torch.randn(*mu.size(), device=mu.device)
        dist_seq = self.fc_seq(seq_feat_r)
        mu_seq = dist_seq[..., :self.bottleneck_bits]
        log_sigma_seq = dist_seq[...,self.bottleneck_bits:]
        epsilon2 = torch.randn(*mu_seq.size(), device=mu_seq.device)
        z_seq = mu_seq + torch.exp(log_sigma_seq / 2) * epsilon2
        kl_seq = 0.5 * torch.mean(torch.exp(log_sigma_seq) + torch.square(mu_seq) - 1. - log_sigma_seq)
        

        #feat_cat = torch.cat((img_feat, seq_feat_cls), -1)
        dist_param = self.fc_fusion(img_feat)

        output = {}
        mu = dist_param[..., :self.bottleneck_bits]
        log_sigma = dist_param[..., self.bottleneck_bits:]
        epsilon = torch.randn(*mu.size(), device=mu.device)
        z = mu + torch.exp(log_sigma / 2) * epsilon
        kl = 0.5 * torch.mean(torch.exp(log_sigma) + torch.square(mu) - 1. - log_sigma)


        if self.mode == 'train':
            img_feat_mod = F.normalize(self.img_mod(z), dim=1)
            img_feat_demod = self.img_demod(img_feat_mod)

            seq_feat_mod = F.normalize(self.seq_mod(z_seq), dim=1)
            seq_feat_demod = self.seq_demod(seq_feat_mod)

        else:
            img_feat_mod = F.normalize(self.img_mod(mu), dim=1)
            img_feat_demod = self.img_demod(img_feat_mod)
            seq_feat_mod = F.normalize(self.seq_mod(mu_seq), dim=1)
            seq_feat_demod = self.seq_demod(seq_feat_mod)

        temp = self.temp.exp()

        if self.skip:
            img_feat_forward = mu
            seq_feat_forward = seq_feat_cls
            output['image_feat_loss'] = torch.tensor([0]).cuda()
            output['seq_feat_loss'] = torch.tensor([0]).cuda()
            output['NCEloss'] = torch.tensor([0]).cuda()
        else:

            image_feat_loss = self.l2loss(z, img_feat_demod)  # featloss
            seq_feat_loss = self.l2loss(z_seq, seq_feat_demod)
            # image @ seq loss  #nceloss
            gt = torch.arange(img_feat.shape[0], dtype=torch.long).to('cuda')
            logit = temp * img_feat_mod @ seq_feat_mod.t()
            cl_loss = self.celoss(logit, gt) + self.celoss(logit.t(), gt)

            output['image_feat_loss'] = image_feat_loss
            output['seq_feat_loss'] = seq_feat_loss
            output['NCEloss'] = cl_loss

            img_feat_forward = img_feat_demod
            seq_feat_forward = seq_feat_demod
        
        #fc_forward = torch.cat((img_feat_forward,seq_feat_forward),-1)
        #fc_forward = self.fc_mergeimgseq(fc_forward)
        fc_img_forward = self.fc_forward_img(img_feat_forward)
        fc_seq_forward = self.fc_forward_seq(seq_feat_forward)



        if self.mode == 'train':
            output['latent']= fc_img_forward

            output['kl_loss'] = kl
            output['kl_seq_loss'] = kl_seq
            seq_feat_[:, 0] = fc_seq_forward
            latent_feat_seq = seq_feat_
            output['seq_latent'] = fc_seq_forward
        else:
            output['latent'] = fc_img_forward
            output['kl_loss'] = 0.0
            output['kl_seq_loss'] = 0.0
            seq_feat_[:, 0] = fc_seq_forward
            latent_feat_seq = seq_feat_
            output['seq_latent'] = fc_seq_forward





        return output, latent_feat_seq