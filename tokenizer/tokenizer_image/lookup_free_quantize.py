from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

from math import sqrt
import math

from einops import rearrange, reduce, pack, unpack

import dist


def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y

def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))

def entropy_loss(
    logits,
    mask=None,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)
    if mask is not None:
        # avg_probs = probs[mask].mean(tuple(range(probs.ndim - 1)))
        # avg_probs = einx.mean("... D -> D", probs[mask])
        avg_probs = reduce(masked_mean(probs, mask), "... D -> D", "mean")
        # avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        # sample_entropy = sample_entropy[mask].mean()
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss



class LFQ(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
            self, codebook_size, Cvae, using_znorm=False, beta: float = 0.25,
            default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,
            num_latent_tokens=256, codebook_drop=0.0, scale=1, 
            sample_minimization_weight=1.0, batch_maximization_weight=1.0, entropy_weight=0.1, soft_entropy=True,
            # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.Cvae: int = Cvae
        self.vocab_size: int = 2 ** self.Cvae
        assert self.vocab_size == codebook_size
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        self.num_latent_tokens = num_latent_tokens
        self.entropy_weight = entropy_weight
        self.soft_entropy = soft_entropy
        self.persample_entropy_compute = 'analytical'

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(share_quant_resi)]))

        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.register_buffer('mask', 2 ** torch.arange(self.Cvae), persistent=False)

        self.beta: float = beta
        
        self.codebook_drop = codebook_drop

        scaler = scale ** torch.arange(len(self.v_patch_nums))
        if using_znorm:
            scaler = scaler / sqrt(self.Cvae)
        
        self.register_buffer('scaler', scaler)
        print("scale is", scaler)
        
        # for entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # codes
        all_codes = torch.arange(codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0

        self.register_buffer('codebook', codebook, persistent = False)

        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1  # progressive training: not supported yet, prog_si always -1

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

        # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False, dropout=None) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        if self.using_znorm: f_BChw = F.normalize(f_BChw, dim=1)
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        # x = f_BChw

        with (torch.cuda.amp.autocast(enabled=False)):
            mean_vq_loss: torch.Tensor = 0.0
            mean_commit_loss: torch.Tensor = 0.0
            mean_entropy_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)

            if self.training:
                max_n = (len(self.v_patch_nums) + 1)
                n_quantizers = torch.ones((B,)) * max_n
                n_dropout = int(B * self.codebook_drop)
                n_quantizers[:n_dropout] = dropout[:n_dropout]
                n_quantizers = n_quantizers.to(f_BChw.device)
            else:
                n_quantizers = torch.ones((B,)) * (self.v_patch_nums + 1)

            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                codebook_value = self.scaler[si].to(device=f_BChw.device, dtype=torch.float).detach()
                # find the nearest embedding
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (
                            si != SN - 1) or pn != int(sqrt(self.num_latent_tokens)) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                # rest_NC = f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                d_no_grad = torch.where(rest_NC > 0, codebook_value, -codebook_value)
                idx_N = self.bits_to_indices((d_no_grad > 0))

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    handler = tdist.all_reduce(hit_V, async_op=True)
                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)

                h_BChw = F.interpolate(self.indices_to_bits(idx_Bhw, si).permute(0, 3, 1, 2), size=(H, W),
                                       mode='bicubic').contiguous() if (si != SN - 1) else self.indices_to_bits(idx_Bhw, si).permute(0, 3, 1, 2).contiguous()
                # h_BChw = self.indices_to_bits(idx_Bhw, si).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)

                # x = f_rest.clone().permute(0, 2, 3, 1)
                x = rearrange((f_BChw - f_hat.detach()), 'b d h w -> b (h w) 1 d')

                mask = (torch.full((B,), fill_value=si, device=h_BChw.device) < n_quantizers)[:, None, None, None].int()
                f_hat = f_hat + h_BChw * mask

                f_rest -= h_BChw
                if self.training:
                    handler.wait()
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                ratio = mask.sum() / B

                codebook = self.codebook * codebook_value

                if self.soft_entropy:
                    per_sample_entropy, codebook_entropy, avg_prob = self.soft_entropy_loss(x, si, codebook, mask.squeeze())
                    entropy_aux_loss = (self.sample_minimization_weight * per_sample_entropy) - (self.batch_maximization_weight * codebook_entropy)
                else:
                    logits = 2 * torch.einsum('... i d, j d -> ... i j', x, codebook)
                    # the same as euclidean distance up to a constant
                    per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                        logits = logits,
                        mask=mask.squeeze(),
                        sample_minimization_weight = self.sample_minimization_weight,
                        batch_maximization_weight = self.batch_maximization_weight
                    )
                # F.mse_loss(f_hat, f_no_grad, reduction="none").mul_(mask).mean() / ratio
                mean_vq_loss += F.mse_loss(f_hat, f_no_grad, reduction="none").mul_(mask).mean() / ratio
                mean_commit_loss += F.mse_loss(f_hat.data, f_BChw, reduction="none").mul_(mask).mul_(self.beta / ratio).mean()

                entropy_weight = self.entropy_weight / ratio
                
                mean_entropy_loss += entropy_aux_loss.mul_(entropy_weight)
                # x -= h_BChw.detach()

            mean_vq_loss *= 1. / SN
            mean_commit_loss *= 1. / SN
            mean_entropy_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in
                      enumerate(self.v_patch_nums)]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss, mean_commit_loss, mean_entropy_loss

    # ===================== `forward` is only used in VAE training =====================

    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.Cvae
        indices = 2 ** torch.arange(
            0,
            self.Cvae,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def indices_to_bits(self, x, si=None):
        """
        x: long tensor of indices

        returns big endian bits
        """
        mask = 2 ** torch.arange(self.Cvae, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        if si == None:
            return x
        return torch.where(x, self.scaler[si], -self.scaler[si])
    
    def soft_entropy_loss(self, z, si, codebook, mask=None):
        if mask != None:
            z = z[mask]
        distance =  - 2 * torch.einsum('... g c, d c ->... g d', z, codebook)  
        prob = (-distance).softmax(dim = -1)
        if self.persample_entropy_compute == 'analytical':
            p = torch.sigmoid(-4 * z * (self.scaler[si]))
            prob = torch.stack([p, 1-p], dim=-1)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean')
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob
    
    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim =True)
        else:
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H


    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[
        List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0],
                                           dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)

        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool,
                           v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[
        Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        if self.using_znorm: f_BChw = F.normalize(f_BChw, dim=1)
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in
                     (v_patch_nums or self.v_patch_nums)]  # from small to large
        # assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            codebook_value = self.scaler[si].to(device=f_BChw.device, dtype=torch.float).detach()
            if 0 <= self.prog_si < si: break  # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (
                        si != SN - 1) or ph != 16 else f_rest.permute(0, 2, 3, 1).reshape(-1, C)

            d_no_grad = torch.where(z_NC > 0, codebook_value, -codebook_value)
            idx_N = self.bits_to_indices((d_no_grad > 0))

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.indices_to_bits(idx_Bhw, si).permute(0, 3, 1, 2), size=(H, W),
                                   mode='bicubic').contiguous() if (si != SN - 1) else self.indices_to_bits(idx_Bhw, si).permute(
                0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))

        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (
                    0 <= self.prog_si - 1 < si): break  # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next),
                                   size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None  # cat BlCs to BLC, this should be float32

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], torch.Tensor]:  # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))  # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]), mode='area')
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

def schedule(ratio, total_unknown, method="cosine"):
    """Generates a mask rate by scheduling mask functions R.

    Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
    training, the input ratio is uniformly sampled; during inference, the input
    ratio is based on the step number divided by the total iteration number: t/T.
    Based on experiements, we find that masking more in training helps.
    Args:
        ratio: The uniformly sampled ratio [0, 1) as input.
        total_unknown: The total number of tokens that can be masked out. For
        example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
        512x512 images.
    method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
        "pow2.5" represents x^2.5

    Returns:
        The mask rate (float).
    """
    if method == "uniform":
        mask_ratio = 1. - ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1. - ratio**exponent
    elif method == "cosine":
        mask_ratio = np.cos(math.pi / 2. * ratio)
    elif method == "log":
        mask_ratio = -np.log2(ratio) / np.log2(total_unknown)
    elif method == "exp":
        mask_ratio = 1 - np.exp2(-np.log2(total_unknown) * (1 - ratio))
    # Clamps mask into [epsilon, 1)
    mask_ratio = np.clip(mask_ratio, 0, 1.)
    return mask_ratio


    
if __name__ == "__main__":

    batch_size = 4
    seq_len = 16
    num_classes = 4096
    # # Generate random logits and integer mask
    # logits = torch.randn(batch_size, seq_len,seq_len, num_classes)
    mask = torch.ones(batch_size, dtype=torch.int) 

    # # Calculate entropy loss
    # sample_entropy, avg_entropy, loss = entropy_loss(
    #     logits,
    #     mask=mask,
    #     sample_minimization_weight=1.0,
    #     batch_maximization_weight=1.0,
    # )

    # # Output results
    # print("Sample Entropy for mask:", sample_entropy)
    # print("Average Entropy for mask:", avg_entropy)
    # print("Entropy Loss for mask:", loss)

    # # Calculate entropy loss
    # sample_entropy, avg_entropy, loss = entropy_loss(
    #     logits,
    #     sample_minimization_weight=1.0,
    #     batch_maximization_weight=1.0,
    # )

    # # Output results
    # print("Sample Entropy:", sample_entropy)
    # print("Average Entropy:", avg_entropy)
    # print("Entropy Loss:", loss)
    quantizer = LFQ(4096, 12, using_znorm=False, v_patch_nums=[1,2,3,4,5,6,8,10,12,16], )

    z = torch.randn(batch_size, seq_len * seq_len, 1, 12)

    for i in range(10):

        codebook = quantizer.codebook * quantizer.scaler[i]
        logits = 2 * torch.einsum('... i d, j d -> ... i j', z, codebook)

        per_sample_entropy, codebook_entropy, avg_prob = quantizer.soft_entropy_loss(z, i, codebook, mask)
        print("Soft Sample Entropy :", per_sample_entropy)
        print("Soft codebook Entropy:", codebook_entropy)
        print("Soft Entropy Loss", per_sample_entropy - codebook_entropy)

        sample_entropy, avg_entropy, loss = entropy_loss(
            logits,
            mask=mask,
            sample_minimization_weight=1.0,
            batch_maximization_weight=1.0,
        )
        print("Sample Entropy :", sample_entropy)
        print("codebook Entropy:", avg_entropy)
        print("Entropy Loss", loss)




    image_feats = torch.randn(2, 12, 16, 16) #16 is dim, must be power of 2 of codebook_size

    dropout_rand = torch.randint(3, len([1,2,3,4,5,6,8,10,12,16]) + 1, (2,))

    quantized, usgae, loss = quantizer(image_feats, ret_usages=True, dropout = dropout_rand)  # you may want to experiment with temperature

    assert image_feats.shape == quantized.shape