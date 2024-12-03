import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import peft
from timm.models import create_model, safe_model_name
from timm.layers import trunc_normal_, Mlp

import sys

from .to_pixel import ToPixel

from .vision_transformer import Attention, RoPEAttention

import math

class DINOv2Encoder(nn.Module):
    def __init__(self, in_channels=3, num_latent_tokens=32, use_attn_mask=False,
                 model_name='vit_small_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0,},
                 pretrained=True, tuning_method='lora', tuning_kwargs={'r': 8}, abs_pos_embed=False, product_quant=1):
        super().__init__()

        assert model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
                              'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
                              'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
                              'vit_large_patch14_reg4_dinov2.lvd142m',
                              'vit_giant_patch14_reg4_dinov2.lvd142m'], f"{model_name} not found"

        # parameters
        self.num_latent_tokens = num_latent_tokens
        self.use_attn_mask = use_attn_mask
        self.product_quant = product_quant

        # load model
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        # model = vit_base_patch14_dinov2(pretrained=pretrained, **model_kwargs)

        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.abs_pos_embed = abs_pos_embed

        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            # lora tuning the backbone
            # config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['norm'], **tuning_kwargs)
            # config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d|.*\.qkv|.*\.proj", modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'lora_unfreeze_patch_embed':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'lat_lora':
            from models.peft_models.lora import LatentLoRALinear
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d|.*\.qkv|.*\.proj", modules_to_save=['norm'],
                                     **tuning_kwargs)
            config._register_custom_module({nn.Linear: LatentLoRALinear})
            self.model = peft.get_peft_model(model, config)
            self.use_attn_mask = True  # force to use attn mask
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        if self.num_latent_tokens:
            # latent tokens
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            nn.init.normal_(self.latent_tokens, std=1e-6)

            if self.abs_pos_embed:
                if self.product_quant > 1:
                    self.lvl_embed = nn.Embedding(1 + self.product_quant, model.embed_dim)
                    patch_size = model_kwargs['patch_size']
                    nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=math.sqrt(1 / model.embed_dim / 3))
                    lvl1LC = torch.cat([torch.full((patch_size * patch_size + 1,), 0),] +
                                        [torch.full((self.num_latent_tokens // self.product_quant,), i + 1)
                                         for i in range(self.product_quant)]).view(1, -1)
                else:
                    self.lvl_embed = nn.Embedding(2, model.embed_dim)
                    patch_size = model_kwargs['patch_size']
                    nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=math.sqrt(1 / model.embed_dim / 3))
                    lvl1LC = torch.cat([torch.full((patch_size * patch_size + 1,), 0),
                                            torch.full((self.num_latent_tokens,), 1)]).view(1, -1)
                self.register_buffer('lvl1LC', lvl1LC)
            else:
                self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
                trunc_normal_(self.latent_pos_embed, std=.02)

            if self.use_attn_mask:
                # create attn mask
                total_length = self.num_img_tokens + self.num_latent_tokens + self.num_prefix_tokens
                attn_mask = torch.zeros((total_length, total_length))
                attn_mask[:self.num_prefix_tokens + self.num_img_tokens, -self.num_latent_tokens:] = -torch.inf
                attn_mask = attn_mask.view(1, 1, total_length, total_length)
                print(attn_mask)
                self.register_buffer('attn_mask', attn_mask)

    def finetine(self, tuning_method, tuning_kwargs={'r': 8}):
        if tuning_method == 'full':
            return
        elif tuning_method == 'lora':
            # lora tuning the backbone
            # config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['norm'], **tuning_kwargs)
            # config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d|.*\.qkv|.*\.proj", modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(self.model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'lora_unfreeze_patch_embed':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(self.model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'lat_lora':
            from models.peft_models.lora import LatentLoRALinear
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d|.*\.qkv|.*\.proj", modules_to_save=['norm'],
                                     **tuning_kwargs)
            config._register_custom_module({nn.Linear: LatentLoRALinear})
            self.model = peft.get_peft_model(self.model, config)
            self.use_attn_mask = True  # force to use attn mask
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in self.model.parameters():
                param.requires_grad = False

    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_tokens', 'latent_pos_embed']

    def forward(self, x, masks=None):

        # get tokens
        x = self.model.patch_embed(x)

        with torch.cuda.amp.autocast(enabled=False):
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)

            if self.num_latent_tokens:
                # insert latent tokens
                z = self.latent_tokens.expand(x.size(0), -1, -1)
                if self.abs_pos_embed:
                    if self.product_quant > 1:
                        H, W = int(math.sqrt(self.num_latent_tokens//self.product_quant)), int(math.sqrt(self.num_latent_tokens//self.product_quant))
                        assert H * W == self.num_latent_tokens // self.product_quant
                        z = z.view(x.size(0), self.product_quant * H, W, -1)
                        z_list = z.chunk(chunks=self.product_quant, dim=1)
                        z_list = [self.model._pos_embed(z)[:, 1:, ] for z in z_list]  # remove cls token
                        x = torch.cat([x,] + z_list, dim=1)
                        x += self.lvl_embed(self.lvl1LC.expand(x.size(0), -1))
                    else:
                        H, W = int(math.sqrt(self.num_latent_tokens)), int(math.sqrt(self.num_latent_tokens))
                        assert H * W == self.num_latent_tokens
                        z = z.view(x.size(0), H, W, -1)
                        z = self.model._pos_embed(z)[:, 1:,]  # remove cls token
                        x = torch.cat([x, z], dim=1)
                        x += self.lvl_embed(self.lvl1LC.expand(x.size(0), -1))
                else:
                    x = torch.cat([x, z + self.latent_pos_embed], dim=1)
        # get dtype
        temp = x.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x = x.to(main_type)

        # pre layer norm
        x = self.model.norm_pre(x)

        # forward backbones
        if self.use_attn_mask:
            for blk in self.model.blocks:
                x = blk(x, self.attn_mask)
        else:
            x = self.model.blocks(x)
        x = self.model.norm(x)

        if self.num_latent_tokens:
            # get z tokens as out
            out = x[:, -self.num_latent_tokens:]
        else:
            # get img tokens as out
            out = x[:, self.num_prefix_tokens:]
        return out


class DINOv2Decoder(nn.Module):
    def __init__(self, in_channels=3,
                 model_name='vit_small_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0}, pretrained=True,
                 tuning_method='lora', tuning_kwargs={'r': 8},
                 num_latent_tokens=32, to_pixel='linear', use_rope=False, cond_latent=False, abs_pos_embed=False):
        super().__init__()

        assert model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
                              'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
                              'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
                              'vit_large_patch14_reg4_dinov2.lvd142m', 'vit_giant_patch14_reg4_dinov2.lvd142m']

        # load model
        if use_rope:
            print("using RoPEAttention")
            attn_layer = RoPEAttention
        else:
            attn_layer = Attention

        model_kwargs['num_latent_tokens'] = num_latent_tokens
        model_kwargs['attn_layer'] = attn_layer
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        self.use_rope = use_rope
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens

        self.abs_pos_embed = abs_pos_embed

        # for n, m in model.named_modules():
        #     print(n, type(m))

        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            # lora tuning the backbone
            # config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False

        # latent tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, self.num_img_tokens, model.embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)
        # self.mask_token = nn.Parameter(torch.clone(model.cls_token))

        if not self.use_rope:
            if self.abs_pos_embed:
                self.lvl_embed = nn.Embedding(2, model.embed_dim)
                patch_size = model_kwargs['patch_size']
                nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=math.sqrt(1 / model.embed_dim / 3))
                lvl1LC = torch.cat([torch.full((patch_size * patch_size + 1,), 0),
                                    torch.full((self.num_latent_tokens + 1,), 1)]).view(1, -1)
                self.register_buffer('lvl1LC', lvl1LC)
            else:
                self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
                trunc_normal_(self.latent_pos_embed, std=.02)
        # from timm.models.vision_transformer import resize_pos_embed
        # latent_pos_embed = resize_pos_embed(model.pos_embed, torch.zeros(1, self.num_latent_tokens, model.embed_dim), 0)
        # self.latent_pos_embed = nn.Parameter(latent_pos_embed)

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_size=model_kwargs['patch_size'])

        # latent initial as pooled dino feature
        self.cond_latent = cond_latent
        if self.cond_latent:
            self.mlp1 = Mlp(model.embed_dim, model.embed_dim, norm_layer=nn.LayerNorm)
            self.mlp2 = Mlp(model.embed_dim, model.embed_dim, norm_layer=nn.LayerNorm)
            self.norm1 = nn.LayerNorm(model.embed_dim)

        del self.model.patch_embed.proj.bias
        del self.model.patch_embed.proj.weight

    
    def finetine(self, tuning_method, tuning_kwargs={'r': 8}):
        if tuning_method == 'full':
            # doing nothing
            return
        elif tuning_method == 'lora':
            # lora tuning the backbone
            # config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(self.model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in self.model.parameters():
                param.requires_grad = False

    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed']

    @property
    def last_layer(self):
        return self.to_pixel.model.weight

    def forward(self, z):

        # mask tokens
        x = self.mask_token.expand(z.size(0), self.num_img_tokens, -1)
        # x = self.mask_token.expand(z.size(0), -1, -1)

        with torch.cuda.amp.autocast(enabled=False):
            if not self.use_rope:
                x = self.model._pos_embed(x)

                if self.cond_latent:
                    ffnout = x + self.mlp1(torch.mean(z.float(), dim=1, keepdim=True))
                    x = x + self.mlp2(self.norm1(ffnout))
                if self.abs_pos_embed:
                    H, W = int(math.sqrt(self.num_latent_tokens)), int(math.sqrt(self.num_latent_tokens))
                    assert H * W == self.num_latent_tokens
                    z = z.view(x.size(0), H, W, -1)
                    z = self.model._pos_embed(z)
                else:
                    z = z + self.latent_pos_embed
            else:
                to_cat = []
                if self.model.cls_token is not None:
                    to_cat.append(self.model.cls_token.expand(x.shape[0], -1, -1))
                if self.model.reg_token is not None:
                    to_cat.append(self.model.reg_token.expand(x.shape[0], -1, -1))
                x = torch.cat(to_cat + [x], dim=1)
            x = self.model.patch_drop(x)

            x = torch.cat([x, z], dim=1)
            if self.abs_pos_embed:
                x += self.lvl_embed(self.lvl1LC.expand(x.size(0), -1))
        # get dtype
        temp = x.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x = x.to(main_type)

        x = self.model.norm_pre(x)

        # forward backbones
        x = self.model.blocks(x)
        x = self.model.norm(x)

        # get img tokens as out
        # x = x[:, z.size(1)+self.num_prefix_tokens:]
        # out = x[:, self.num_prefix_tokens:]
        # x = x[:, -self.num_img_tokens:]
        # x = self.to_pixel(x)
        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        out = self.to_pixel(x)

        return out


class DINOv2Decoder_(nn.Module):
    def __init__(self, in_channels=3,
                 model_name='vit_small_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0}, pretrained=True,
                 tuning_method='lora', tuning_kwargs={'r': 8}, to_pixel='linear', use_rope=False, cond_latent=False):
        super().__init__()

        assert model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
                              'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
                              'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
                              'vit_large_patch14_reg4_dinov2.lvd142m', 'vit_giant_patch14_reg4_dinov2.lvd142m']

        # load model
        if use_rope:
            print("using RoPEAttention")
            attn_layer = RoPEAttention
        else:
            attn_layer = Attention

        model_kwargs['attn_layer'] = attn_layer
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        self.use_rope = use_rope
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens

        # for n, m in model.named_modules():
        #     print(n, type(m))

        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            # lora tuning the backbone
            # config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False

        # from timm.models.vision_transformer import resize_pos_embed
        # latent_pos_embed = resize_pos_embed(model.pos_embed, torch.zeros(1, self.num_latent_tokens, model.embed_dim), 0)
        # self.latent_pos_embed = nn.Parameter(latent_pos_embed)

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_size=model_kwargs['patch_size'])

        # latent initial as pooled dino feature
        self.cond_latent = cond_latent
        if self.cond_latent:
            self.mlp1 = Mlp(model.embed_dim, model.embed_dim, norm_layer=nn.LayerNorm)
            self.mlp2 = Mlp(model.embed_dim, model.embed_dim, norm_layer=nn.LayerNorm)
            self.norm1 = nn.LayerNorm(model.embed_dim)

    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed']

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=False):
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)
        # get dtype
        temp = x.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x = x.to(main_type)

        x = self.model.norm_pre(x)

        # forward backbones
        x = self.model.blocks(x)
        x = self.model.norm(x)

        # get img tokens as out
        # x = x[:, z.size(1)+self.num_prefix_tokens:]
        # out = x[:, self.num_prefix_tokens:]
        # x = x[:, -self.num_img_tokens:]
        # x = self.to_pixel(x)
        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        out = self.to_pixel(x)

        return out


if __name__ == '__main__':
    encoder = DINOv2Encoder(model_name='vit_small_patch14_dinov2.lvd142m',
                            model_kwargs={'img_size': 256, 'patch_size': 16, 'drop_path_rate': 0.0},
                            tuning_method='lat_lora', tuning_kwargs={'r': 8},
                            num_latent_tokens=32)
    decoder = DINOv2Decoder(model_name='vit_small_patch14_dinov2.lvd142m',
                            model_kwargs={'img_size': 256, 'patch_size': 16, 'drop_path_rate': 0.0},
                            tuning_method='full', tuning_kwargs={'r': 8}, num_latent_tokens=32, use_rope=True)
    x = torch.randn(1, 3, 256, 256)
    out = encoder(x)
    out = decoder(out)
    print(out.shape)