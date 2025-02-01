# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.vqgan.cliploss import ClipLoss
from timm.models import create_model

import sys, os
from math import sqrt
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

sys.path.append(project_root)

from tokenizer.tokenizer_image.quant import VectorQuantizer2
from tokenizer.tokenizer_image.lookup_free_quantize import LFQ
from tokenizer.tokenizer_image.dino_enc.dinov2 import DINOv2Encoder, DINOv2Decoder
from datasets import Denormalize
from datasets import Normalize as ImgNormalize

import torch.distributed as tdist

@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    v_patch_nums: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    semantic_guide: str = 'dinov2'
    detail_guide: str = 'clip'
    num_latent_tokens: int = 256
    encoder_model: str = 'vit_small_patch14_dinov2.lvd142m'
    decoder_model: str = 'vit_small_patch14_dinov2.lvd142m'
    abs_pos_embed: bool = False
    share_quant_resi: int = 4
    product_quant: int = 1
    codebook_drop: float = 0.0
    half_sem: bool = False
    start_drop: int = 1
    sem_loss_weight: float = 0.1
    detail_loss_weight: float = 0.1
    clip_norm: bool = False
    sem_loss_scale: float = 1.0
    detail_loss_scale: float = 1.0
    guide_type_1: str = "class"
    guide_type_2: str = "class"

    lfq: bool = False
    scale: float = 1.0
    soft_entropy: bool = True

    dependency_loss_weight: float = 0.0

    test_model: bool = False


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs,):
        super().__init__()
        self.config = config
        self.enc_type = config.enc_type
        self.dec_type = config.dec_type
        self.product_quant = config.product_quant
        self.half_sem = config.half_sem
        self.start_drop = config.start_drop
        self.clip_norm = config.clip_norm
        config.num_latent_tokens = config.num_latent_tokens * config.product_quant  # scale num_latent_tokens for PQ

        if config.enc_type == 'cnn':
            self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'dinov2':
            self.encoder = DINOv2Encoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': 256, 'patch_size': 16, 'drop_path_rate': 0.1},
                pretrained=True,
                tuning_method='full',
                tuning_kwargs={'r':8},
                abs_pos_embed=config.abs_pos_embed,
                product_quant=config.product_quant,
            )
            self.quant_conv = nn.Conv2d(self.encoder.embed_dim, config.codebook_embed_dim, 1)
        else:
            raise NotImplementedError

        if config.dec_type == 'cnn':
            self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'dinov2':
            self.decoder = DINOv2Decoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens // self.product_quant,
                model_name=config.decoder_model,
                model_kwargs={'img_size': 256, 'patch_size': 16, 'drop_path_rate': 0.1},
                pretrained=True,
                tuning_method='full',
                tuning_kwargs={'r':8},
                to_pixel='linear',
                use_rope=False,
                cond_latent=False,
                abs_pos_embed=config.abs_pos_embed,
            )
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.decoder.embed_dim, 1)

        self.V = self.vocab_size = config.codebook_size * self.product_quant
        self.Cvae = config.codebook_embed_dim * self.product_quant
        if self.product_quant > 1:
            if len(config.v_patch_nums) == 1:
                self.quantizes = nn.ModuleList([VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                                                config.commit_loss_beta, config.codebook_l2_norm) for _ in range(self.product_quant)])
            elif not config.lfq:
                self.quantizes = nn.ModuleList([VectorQuantizer2(config.codebook_size, config.codebook_embed_dim,
                                                 v_patch_nums=config.v_patch_nums,
                                                 num_latent_tokens=config.num_latent_tokens // self.product_quant,
                                                 share_quant_resi=config.share_quant_resi,
                                                 codebook_drop=config.codebook_drop,) for _ in range(self.product_quant)])
            else:
                self.quantizes = nn.ModuleList([LFQ(config.codebook_size, config.codebook_embed_dim,
                                                v_patch_nums=config.v_patch_nums,
                                                num_latent_tokens=config.num_latent_tokens // self.product_quant,
                                                share_quant_resi=config.share_quant_resi,
                                                codebook_drop=config.codebook_drop,
                                                using_znorm=config.codebook_l2_norm,
                                                scale=config.scale,
                                                entropy_weight=config.entropy_loss_ratio,
                                                soft_entropy=config.soft_entropy,
                                                ) for _ in range(self.product_quant)])
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim * self.product_quant, self.decoder.embed_dim, 1)
        else:
            if len(config.v_patch_nums) == 1:
                self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, config.commit_loss_beta, config.codebook_l2_norm)
            elif not config.lfq:
                self.quantize = VectorQuantizer2(config.codebook_size, config.codebook_embed_dim,
                                                 v_patch_nums=config.v_patch_nums,
                                                 num_latent_tokens=config.num_latent_tokens,
                                                 share_quant_resi=config.share_quant_resi,
                                                 )
            else:
                self.quantize = LFQ(config.codebook_size, config.codebook_embed_dim,
                                    v_patch_nums=config.v_patch_nums,
                                    num_latent_tokens=config.num_latent_tokens,
                                    share_quant_resi=config.share_quant_resi,
                                    codebook_drop=config.codebook_drop,
                                    using_znorm=config.codebook_l2_norm,
                                    scale=config.scale,
                                    entropy_weight=config.entropy_loss_ratio,
                                    soft_entropy=config.soft_entropy)

        self.codebook_embed_dim = config.codebook_embed_dim
        self.v_patch_nums = config.v_patch_nums
        self.codebook_drop = config.codebook_drop
        # Semantic loss to preserve dino semantics
        self.semantic_guide = config.semantic_guide
        self.denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalize = ImgNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.semantic_guide == 'dinov2':
            semantic_model = create_model(config.encoder_model, pretrained=True, img_size=256, patch_size=16,
                                          drop_path_rate=0.0)
            semantic_model.eval()
            for param in semantic_model.parameters():
                param.requires_grad = False
            self.semantic_model = semantic_model # torch.compile(semantic_model, mode='max-autotune')

            local_loss = False
            gather_with_grad = True
            rank = tdist.get_rank()
            world_size = tdist.get_world_size()
            use_horovod = False
            sem_loss_scale = config.sem_loss_scale

            self.sem_loss_scale = sem_loss_scale
            self.semantic_loss = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=True,
                rank=rank,
                world_size=world_size,
                use_horovod=use_horovod,
            )
            if not self.half_sem:
                self.sem_linear = nn.Conv2d(self.product_quant * config.codebook_embed_dim, config.codebook_embed_dim, 1)
            if self.enc_type == 'cnn':
                self.sem_linear = torch.nn.Linear(384, config.codebook_embed_dim)

            self.sem_loss_weight = config.sem_loss_weight
        
        self.detail_guide = config.detail_guide
        if self.detail_guide != 'none':
            detail_model = create_model("vit_base_patch16_clip_224.openai", pretrained=True, img_size=256, patch_size=16,
                                          drop_path_rate=0.0)
            detail_model.eval()
            for param in detail_model.parameters():
                param.requires_grad = False
            self.detail_model = detail_model

            self.detail_loss_scale = config.detail_loss_scale
            self.detail_loss = ClipLoss(
                local_loss=False,
                gather_with_grad=True,
                cache_labels=True,
                rank=tdist.get_rank(),
                world_size=tdist.get_world_size(),
                use_horovod=False
            )
            self.detail_loss_weight = config.detail_loss_weight
        
        self.guide_type_1 = config.guide_type_1
        self.guide_type_2 = config.guide_type_2
        self.dependency_loss_weight = config.dependency_loss_weight

        self.test_mode = config.test_model

        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    def finetune(self, enc_tuning_method, dec_tuning_method):
        self.encoder.finetine(enc_tuning_method)
        self.decoder.finetine(dec_tuning_method)

    def encode(self, x):
        h = self.encoder(x)
        if self.enc_type == 'dinov2':
            b, l, c = h.shape
            if self.product_quant > 1:
                assert int(sqrt(l//self.product_quant)) ** 2 * self.product_quant == l
                h = h.view(b, l, 1, c)
                h = h.permute(0, 3, 1, 2)
            else:
                assert int(sqrt(l)) ** 2 == l
                h = h.view(b, int(sqrt(l)), int(sqrt(l)), c)
                h = h.permute(0, 3, 1, 2)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, return_quant=False):
        quant = self.post_quant_conv(quant)
        if self.dec_type == 'dinov2':
            quant = quant.flatten(2).permute(0, 2, 1)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b,):
        quant_b, usages, mean_vq_loss = self.quantize(code_b, ret_usages=True)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, epoch):
        h = self.encode(input)
        b, c, l, _ = h.shape
        if len(self.v_patch_nums) == 1:
            dropout_rand = None
        else:
            dropout_rand = torch.randint(self.start_drop, len(self.v_patch_nums) + 1, (b,))  # to fix dropout across quantizers, skip first start_drop-1 quantizers

        if self.product_quant > 1:
            h_list = h.chunk(chunks=self.product_quant, dim=2)
            quant_list, usages_list, mean_vq_loss_list, commit_loss_list, entropy_list = [], [], [], [], []
            for i, h in enumerate(h_list):
                h = h.view(b, -1, int(sqrt(l // self.product_quant)), int(sqrt(l // self.product_quant)))
                quant, usages, vq_loss, commit_loss, entropy_loss = self.quantizes[i].forward(h, ret_usages=True, dropout=dropout_rand)
                quant_list.append(quant)
                usages_list.append(usages)
                mean_vq_loss_list.append(vq_loss)
                commit_loss_list.append(commit_loss)
                entropy_list.append(entropy_loss)
            dependency_loss = self.dependency_loss_weight * orthogonal_cosine_loss(torch.mean(quant_list[0], dim=(2, 3)).contiguous(), torch.mean(quant_list[-1], dim=(2, 3)).contiguous())
            usages = [sum(us) / self.product_quant for us in zip(*usages_list)]
            mean_vq_loss = sum(mean_vq_loss_list) / self.product_quant
            mean_commit_loss = sum(commit_loss_list) / self.product_quant
            mean_entropy = sum(entropy_list) / self.product_quant
            quant = torch.cat(quant_list, dim=1)
        else:
            dependency_loss = 0.0
            quant, usages, mean_vq_loss, mean_commit_loss, mean_entropy = self.quantize.forward(h, ret_usages=True, dropout=dropout_rand)
            quant_list = [quant]


        dec = self.decode(quant)

        # normalize the inputs to dino's transform
        input = self.normalize(self.denormalize(input))
        if self.semantic_guide != 'none':
            if self.guide_type_1 == 'class':
                z_s = self.semantic_model(input)
                z_s = z_s[..., None, None]
            else:
                z_s = self.semantic_model.forward_features(input)[:, 1:, :]
                z_s = z_s.reshape(b, 768, 16, 16)
            if self.enc_type == 'dinov2':
                z_s = self.quant_conv(z_s).contiguous()
                semantic_quant = quant_list[-1]
                z_s = torch.mean(z_s, dim=(2, 3)).contiguous()
                z_q_ = torch.mean(semantic_quant, dim=(2, 3)).contiguous()
            elif self.enc_type == 'cnn':
                z_q_ = torch.mean(h, dim=(2, 3)).contiguous()
                z_s = self.sem_linear(z_s).contiguous()

            n_drop = int(b * self.codebook_drop)
            with (torch.cuda.amp.autocast(enabled=False)):
                sem_loss_scale = self.sem_loss_scale
                feat1 = z_s[n_drop:].float()
                feat2 = z_q_[n_drop:].float()
                if self.clip_norm:
                    feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
                    feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
                    sem_loss_scale = (epoch % 200) / 200 * (100 - sem_loss_scale) + sem_loss_scale if epoch < 200 else 100
                sem_loss = self.semantic_loss.forward(feat1, feat2, logit_scale=sem_loss_scale)
                sem_loss = sem_loss * self.sem_loss_weight
        else:
            sem_loss = None
        
        if self.detail_guide != 'none':
            assert self.guide_type_2 == 'patch', "current only accept patch for detail guide"
            if self.guide_type_2 == 'class':
                z_d = self.detail_model(input)
                z_d = z_d[..., None, None]
            else:
                z_d = self.detail_model.forward_features(input)[:, 1:, :]
                z_d = z_d.reshape(b, 768, 16, 16)
            if self.enc_type == 'dinov2':
                z_d = self.quant_conv(z_d).contiguous()
                detail_quant = quant_list[0]
                z_d = torch.mean(z_d, dim=(2, 3)).contiguous()
                z_q_ = torch.mean(detail_quant, dim=(2, 3)).contiguous()
            elif self.enc_type == 'cnn':
                pass
            
            n_drop = int(b * self.codebook_drop)
            with (torch.cuda.amp.autocast(enabled=False)):
                detail_loss_scale = self.detail_loss_scale
                feat1 = z_s[n_drop:].float()
                feat2 = z_q_[n_drop:].float()
                if self.clip_norm:
                    feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
                    feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
                    detail_loss_scale = (epoch % 200) / 200 * (100 - detail_loss_scale) + detail_loss_scale if epoch < 200 else 100
                detail_loss = self.detail_loss.forward(feat1, feat2, logit_scale=detail_loss_scale)
                detail_loss = detail_loss * self.detail_loss_weight
        else:
            detail_loss = None

        return dec, (mean_vq_loss, mean_commit_loss, mean_entropy, usages), sem_loss, detail_loss, dependency_loss

    def img_to_reconstructed_img(self, x, last_one=True,) -> List[torch.Tensor]:
        h = self.encoder(x)
        if self.enc_type == 'dinov2':
            b, l, c = h.shape
            if self.product_quant > 1:
                assert int(sqrt(l // self.product_quant)) ** 2 * self.product_quant == l
                h = h.view(b, l, 1, c)
                h = h.permute(0, 3, 1, 2)
            else:
                assert int(sqrt(l)) ** 2 == l
                h = h.view(b, int(sqrt(l)), int(sqrt(l)), c)
                h = h.permute(0, 3, 1, 2)
        f = self.quant_conv(h)

        if self.product_quant > 1:
            b, c, l, _ = f.shape
            f_list = f.chunk(chunks=self.product_quant, dim=2)
            f_list = [f.view(b, -1, int(sqrt(l // self.product_quant)), int(sqrt(l // self.product_quant))) for f in f_list]
            if len(self.v_patch_nums) == 1:
                f_hats_list = [self.quantizes[i].f_to_idxBl_or_fhat(f, to_fhat=True) for i, f in enumerate(f_list)]
            else:
                f_hats_list = [self.quantizes[i].f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=self.v_patch_nums) for i, f in enumerate(f_list)]
            f_hats = [self.post_quant_conv(torch.cat(f_hats, dim=1)) for f_hats in zip(*f_hats_list)]
        else:
            if len(self.v_patch_nums) == 1:
                ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=None)
            else:
                ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=self.v_patch_nums)
            f_hats = [self.post_quant_conv(f_hat) for f_hat in ls_f_hat_BChw]

        if self.dec_type == 'dinov2':
            f_hats = [f_hat.flatten(2).permute(0, 2, 1) for f_hat in f_hats]

        if last_one:
            return self.decoder(f_hats[-1]).clamp_(-1, 1)
        else:
            return [self.decoder(f_hat).clamp_(-1, 1) for f_hat in f_hats]

    def img_to_sem_feat(self, x,) -> List[torch.Tensor]:
        h = self.encoder(x)
        if self.enc_type == 'dinov2':
            b, l, c = h.shape
            if self.product_quant > 1:
                assert int(sqrt(l // self.product_quant)) ** 2 * self.product_quant == l
                h = h.view(b, l, 1, c)
                h = h.permute(0, 3, 1, 2)
            else:
                assert int(sqrt(l)) ** 2 == l
                h = h.view(b, int(sqrt(l)), int(sqrt(l)), c)
                h = h.permute(0, 3, 1, 2)
        f = self.quant_conv(h)


        b, c, l, _ = f.shape
        f_list = f.chunk(chunks=self.product_quant, dim=2)
        f_list = [f.view(b, -1, int(sqrt(l // self.product_quant)), int(sqrt(l // self.product_quant))) for f in f_list]
        f_hats_list = [self.quantizes[i].f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=self.v_patch_nums) for i, f in enumerate(f_list)]

        z_q = f_hats_list[-1][-1] # torch.mean(f_hats_list[-1][-1], dim=(2, 3)).contiguous()
        return z_q
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        f_hat = self.post_quant_conv(f_hat)
        if self.dec_type == 'dinov2':
            f_hat = f_hat.flatten(2).permute(0, 2, 1)
        return self.decoder(f_hat).clamp_(-1, 1)
    
    def idxBl_to_var_input(self, gt_idx_Bl):
        if self.product_quant > 1:
            x_BLCv_wo_first_l_list = [self.quantizes[i].idxBl_to_var_input(gt_idx_Bl[i]) for i in range(self.product_quant)]
            return torch.cat(x_BLCv_wo_first_l_list, dim=-1)
        else:
            return self.quantize.idxBl_to_var_input(gt_idx_Bl)
    
    def get_next_autoregressive_input(self, si, SN, f_hat, h_BChw):
        f_hat_list = f_hat.chunk(self.product_quant, dim=1)
        h_BChw_list = h_BChw.chunk(self.product_quant, dim=1)
        out_fhat_list, out_next_token_map_list = [], []
        for i, (f_hat, h_BChw) in enumerate(zip(f_hat_list, h_BChw_list)):
            out_fhat, out_next_token_map = self.quantizes[i].get_next_autoregressive_input(si, SN, f_hat, h_BChw)
            out_fhat_list.append(out_fhat)
            out_next_token_map_list.append(out_next_token_map)
        f_hat = torch.cat(out_fhat_list, dim=1)
        next_token_map = torch.cat(out_next_token_map_list, dim=1)
        return f_hat, next_token_map


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

class VectorQuantizer(nn.Module):

    def __init__(self, vocab_size=8192, z_channels=32, beta=0.25, codebook_norm=True):
        super().__init__()
        # parameters
        self.vocab_size = vocab_size
        self.z_channels = z_channels
        self.beta = beta
        self.codebook_norm = codebook_norm
        # self.restart_unused_codes = restart_unused_codes

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.z_channels)
        self.embedding.weight.data.uniform_(-1.0 / self.vocab_size, 1.0 / self.vocab_size)
        if self.codebook_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        
        self.register_buffer("ema_vocab_hit_SV", torch.full((self.vocab_size,), fill_value=0.0))
        self.record_hit = 0

    def no_weight_decay(self):
        return ['embedding.weight',]

    def forward(self, z, ret_usages=True, dropout=None):

        vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=z.device)

        # reshape z -> (batch, height * width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.z_channels)

        if self.codebook_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        # argmin find indices and embeddings
        min_encoding_indices = torch.argmin(d, dim=1)
        
        
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.codebook_norm:
            z_q = F.normalize(z_q, p=2, dim=-1)

        if ret_usages and self.training:
            hit_V = min_encoding_indices.bincount(minlength=self.vocab_size).float()
            handler = tdist.all_reduce(hit_V, async_op=True)
            handler.wait()
            if self.record_hit == 0:
                self.ema_vocab_hit_SV.copy_(hit_V)
            elif self.record_hit < 100:
                self.ema_vocab_hit_SV.mul_(0.9).add_(hit_V.mul(0.1))
            else:
                self.ema_vocab_hit_SV.mul_(0.99).add_(hit_V.mul(0.01))
            self.record_hit += 1
            vocab_hit_V.add_(hit_V)

            margin = tdist.get_world_size() * (z.numel() / self.z_channels) / self.vocab_size * 0.08

            codebook_usage = (self.ema_vocab_hit_SV >= margin).float().mean().item() * 100


        # compute loss
        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
        vq_loss = torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients - "straight-through"
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, [codebook_usage], vq_loss, commit_loss, 0.0
    
    def f_to_idxBl_or_fhat(self, z: torch.Tensor, to_fhat: bool, v_patch_nums):  # z_BChw is the feature from inp_img_no_grad
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.z_channels)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.codebook_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        # argmin find indices and embeddings
        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.codebook_norm:
            z_q = F.normalize(z_q, p=2, dim=-1)

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        f_hat_or_idx_Bl: List[torch.Tensor] = [z_q if to_fhat else min_encoding_indices]

        return f_hat_or_idx_Bl


def orthogonal_cosine_loss(A, B):
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)
    loss = (A_norm * B_norm).sum(dim=1).mean()
    return loss

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}

if __name__ == '__main__':
    semantic_model = create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, img_size=256, patch_size=16,
                                  drop_path_rate=0.0)
    semantic_model.eval()

