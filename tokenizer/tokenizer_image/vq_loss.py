# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator
from tokenizer.tokenizer_image.discriminator_dino import DinoDisc as DINODiscriminator
from tokenizer.tokenizer_image.diffaug import DiffAug
import wandb
import torch.distributed as tdist

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def anneal_weight(weight, global_step, threshold=0, initial_value=0.3, final_value=0.1, anneal_steps=2000):
    if global_step < threshold:
        return initial_value
    elif global_step < threshold + anneal_steps:
        # Linearly interpolate between initial and final values within the anneal_steps
        decay_ratio = (global_step - threshold) / anneal_steps
        weight = initial_value - decay_ratio * (initial_value - final_value)
    else:
        # After annealing steps, set to final value
        weight = final_value
    return weight

class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg

class VQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, lecam_loss_weight=None, norm_type='bn', aug_prob=1,
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan", 'dinodisc', 'samdisc']
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        self.disc_type = disc_type
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        elif disc_type == "dinodisc":
            self.discriminator = DINODiscriminator(norm_type=norm_type)  # default 224 otherwise crop
            self.daug = DiffAug(prob=aug_prob, cutout=0.2)
        elif disc_type == "samdisc":
            self.discriminator = SAMDiscriminator(norm_type=norm_type)
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()

        if tdist.get_rank() == 0:
            self.wandb_tracker = wandb.init(project='MSVQ',)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, codebook_loss, sem_loss, detail_loss, dependency_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None,
                logger=None, log_every=100, fade_blur_schedule=0):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)

            # discriminator loss
            if self.disc_type == "dinodisc":
                if fade_blur_schedule < 1e-6:
                    fade_blur_schedule = 0
                logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous(), fade_blur_schedule))
            else:
                logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            if sem_loss is None:
                sem_loss = 0
            if detail_loss is None:
                detail_loss = 0
            if dependency_loss is None:
                dependency_loss = 0
            loss = self.rec_weight * rec_loss + \
                self.perceptual_weight * p_loss + \
                disc_adaptive_weight * disc_weight * generator_adv_loss + \
                codebook_loss[0] + codebook_loss[1] + codebook_loss[2] + sem_loss + detail_loss + dependency_loss
            
            if global_step % log_every == 0:
                rec_loss = self.rec_weight * rec_loss
                p_loss = self.perceptual_weight * p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, sem_loss: {sem_loss:.4f}, detail_loss: {detail_loss} "
                            f"dependency_loss: {dependency_loss} vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
                            f"codebook_usage: {codebook_loss[3]}, generator_adv_loss: {generator_adv_loss:.4f}, "
                            f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
                if tdist.get_rank() == 0:
                    self.wandb_tracker.log({
                        "rec_loss": rec_loss,
                        "perceptual_loss": p_loss,
                        "sem_loss": sem_loss,
                        "detail_loss": detail_loss,
                        "dependency_loss": dependency_loss, 
                        "vq_loss": codebook_loss[0],
                        "commit_loss": codebook_loss[1],
                        "entropy_loss": codebook_loss[2],
                        "codebook_usage": np.mean(codebook_loss[3]),
                        "generator_adv_loss": generator_adv_loss,
                        "disc_adaptive_weight": disc_adaptive_weight,
                        "disc_weight": disc_weight,
                    },
                    step=global_step)
            return loss

        # discriminator update
        if optimizer_idx == 1:

            if self.disc_type == "dinodisc":
                if fade_blur_schedule < 1e-6:
                    fade_blur_schedule = 0
                # add blur since disc is too strong
                logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous().detach(), fade_blur_schedule))
                logits_real = self.discriminator(self.daug.aug(inputs.contiguous().detach(), fade_blur_schedule))
            else:
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                logits_real = self.discriminator(inputs.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

            if self.lecam_loss_weight is not None:
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_adversarial_loss = disc_weight * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
            
            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                logger.info(f"(Discriminator) " 
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}")
                if tdist.get_rank() == 0:
                    self.wandb_tracker.log({
                        "discriminator_adv_loss": d_adversarial_loss,
                        "disc_weight": disc_weight,
                        "logits_real": logits_real,
                        "logits_fake": logits_fake,
                    },
                    step=global_step)
            return d_adversarial_loss