# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from huggingface_hub import upload_folder

import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from tqdm import tqdm
import ruamel.yaml as yaml

import os
import time
import argparse
from glob import glob
from copy import deepcopy
import sys
import math
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr, center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.xqgan_model import VQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss

from timm.scheduler import create_scheduler_v2 as create_scheduler

from evaluator import Evaluator
import tensorflow.compat.v1 as tf

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import warnings
warnings.filterwarnings('ignore')

import wandb
#################################################################################
#                                  Training Loop                                #
#################################################################################

def get_random_ratio(randomness_anneal_start, randomness_anneal_end, end_ratio, cur_step):
    if cur_step < randomness_anneal_start:
        return 1.0
    elif cur_step > randomness_anneal_end:
        return end_ratio
    else:
        return 1.0 - (cur_step - randomness_anneal_start) / (randomness_anneal_end - randomness_anneal_start) * end_ratio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/mnt/localssd/ImageNet2012/train')
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, default='output/debug', help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-epoch-start", type=int, default=0, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-start", type=int, default=0, help="iteration to start discriminator training and loss")  # autoset
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--disc-weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--save_best",action='store_true', default=False)
    parser.add_argument("--val_data_path", type=str, default="/mnt/localssd/ImageNet2012/val")
    parser.add_argument("--sample_folder_dir", type=str, default='samples')
    parser.add_argument("--reconstruction_folder_dir", type=str, default='reconstruction')
    parser.add_argument("--v-patch-nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], nargs='+',
                        help="number of patch numbers of each scale")
    parser.add_argument("--enc_type", type=str, default="cnn")
    parser.add_argument("--dec_type", type=str, default="cnn")
    parser.add_argument("--semantic_guide", type=str, default="none")
    parser.add_argument("--detail_guide", type=str, default="none")
    parser.add_argument("--num_latent_tokens", type=int, default=256)
    parser.add_argument("--encoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='encoder model name')
    parser.add_argument("--decoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='encoder model name')
    parser.add_argument("--disc_adaptive_weight", type=bool, default=False)
    parser.add_argument("--abs_pos_embed", type=bool, default=False)
    parser.add_argument("--product_quant", type=int, default=1)
    parser.add_argument("--share_quant_resi", type=int, default=4)
    parser.add_argument("--codebook_drop", type=float, default=0.0)
    parser.add_argument("--half_sem", type=bool, default=False)
    parser.add_argument("--start_drop", type=int, default=1)
    parser.add_argument("--lecam_loss_weight", type=float, default=None)
    parser.add_argument("--sem_loss_weight", type=float, default=0.1)
    parser.add_argument("--detail_loss_weight", type=float, default=0.1)
    parser.add_argument("--enc_tuning_method", type=str, default='full')
    parser.add_argument("--dec_tuning_method", type=str, default='full')
    parser.add_argument("--clip_norm", type=bool, default=False)
    parser.add_argument("--sem_loss_scale", type=float, default=1.0)
    parser.add_argument("--detail_loss_scale", type=float, default=1.0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--norm_type", type=str, default='bn')
    parser.add_argument("--aug_prob", type=float, default=1.0)
    parser.add_argument("--aug_fade_steps", type=int, default=0)
    parser.add_argument("--disc_reinit", type=int, default=0)
    parser.add_argument("--debug_disc", type=bool, default=False)
    parser.add_argument("--guide_type_1", type=str, default='class', choices=["patch", "class"])
    parser.add_argument("--guide_type_2", type=str, default='class', choices=["patch", "class"])
    parser.add_argument("--lfq", action='store_true', default=False, help="if use LFQ")

    parser.add_argument("--end-ratio", type=float, default=0.5)
    parser.add_argument("--anneal-start", type=int, default=200)
    parser.add_argument("--anneal-end", type=int, default=200)
    
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--delta", type=int, default=100)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

        # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        cloud_results_dir = f"{args.cloud_save_path}"
        cloud_checkpoint_dir = f"{cloud_results_dir}"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")

        experiment_config = vars(args)
        with open(os.path.join(cloud_checkpoint_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    if args.save_best:
        transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        args.data_path = args.val_data_path
        val_dataset = build_dataset(args, transform=transform)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        if rank % torch.cuda.device_count() == 0:
            os.makedirs(args.sample_folder_dir, exist_ok=True)
            os.makedirs(args.reconstruction_folder_dir, exist_ok=True)
            logger.info(f"Saving .png samples at {args.sample_folder_dir}")
            logger.info(f"Saving .png reconstruction at {args.reconstruction_folder_dir}")

    num_update_steps_per_epoch = len(loader)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    args.disc_start = args.disc_epoch_start * num_update_steps_per_epoch

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        v_patch_nums=args.v_patch_nums,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        semantic_guide=args.semantic_guide,
        detail_guide=args.detail_guide,
        num_latent_tokens=args.num_latent_tokens,
        abs_pos_embed=args.abs_pos_embed,
        share_quant_resi=args.share_quant_resi,
        product_quant=args.product_quant,
        codebook_drop=args.codebook_drop,
        half_sem=args.half_sem,
        start_drop=args.start_drop,
        sem_loss_weight=args.sem_loss_weight,
        detail_loss_weight=args.detail_loss_weight,
        clip_norm=args.clip_norm,
        sem_loss_scale=args.sem_loss_scale,
        detail_loss_scale=args.detail_loss_scale,
        guide_type_1=args.guide_type_1,
        guide_type_2=args.guide_type_2,
        lfq=args.lfq
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)
    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        lecam_loss_weight=args.lecam_loss_weight,
        disc_adaptive_weight=args.disc_adaptive_weight,
        norm_type=args.norm_type,
        aug_prob=args.aug_prob,
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    args.lr = args.lr * args.global_batch_size / 128
    args.disc_lr = args.disc_lr * args.global_batch_size / 128
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    optimizer = torch.optim.AdamW(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay, )
    optimizer_disc = torch.optim.AdamW(vq_loss.discriminator.parameters(), lr=args.disc_lr,
                                       betas=(args.beta1, args.beta2), weight_decay=args.disc_weight_decay, )

    # create lr scheduler
    if args.lr_scheduler == 'none':
        vqvae_lr_scheduler = None
        disc_lr_scheduler = None
    else:
        vqvae_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer,
            patience_epochs=0,
            step_on_epochs=True,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=1,
            min_lr=5e-5,
        )
        disc_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer_disc,
            patience_epochs=0,
            step_on_epochs=True,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs - args.disc_epoch_start,
            warmup_epochs=int(0.02 * args.epochs),
            min_lr=5e-5,
        )

    logger.info(f"num_update_steps_per_epoch {num_update_steps_per_epoch:,} max_train_steps ({max_train_steps})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not args.debug_disc:
            vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        else:
            num_step = checkpoint["optimizer_disc"]["state"][next(iter(checkpoint["optimizer_disc"]["state"]))]['step']
            for param_state in optimizer_disc.state.values():
                param_state['step'] = num_step
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size)) + 1
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        vq_model.finetune(args.enc_tuning_method, args.dec_tuning_method)
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model, mode='max-autotune')  # requires PyTorch 2.0
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu])
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    curr_fid = None

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        ratio = get_random_ratio(args.anneal_start, args.anneal_end, args.end_ratio, epoch)
        delta = int(ratio * args.delta)
        alpha = ratio * args.alpha
        beta = args.beta

        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        if args.disc_reinit != 0:
            if epoch % args.disc_reinit == 0:
                vq_loss.module.discriminator.reinit()
        for x, y in loader:
            imgs = x.to(device, non_blocking=True)

            if args.aug_fade_steps >= 0:
                fade_blur_schedule = 0 if train_steps < args.disc_start else min(1.0, (train_steps - args.disc_start) / (args.aug_fade_steps + 1))
                fade_blur_schedule = 1 - fade_blur_schedule
            else:
                fade_blur_schedule = 0
            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, codebook_loss, sem_loss, detail_loss, dependency_loss = vq_model(imgs, epoch, alpha, beta, delta)
                loss_gen = vq_loss(codebook_loss, sem_loss, detail_loss, dependency_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1,
                                   last_layer=vq_model.module.decoder.last_layer, 
                                   logger=logger, log_every=args.log_every, fade_blur_schedule=fade_blur_schedule)

            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # discriminator training            
            optimizer_disc.zero_grad()

            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(codebook_loss, sem_loss, detail_loss, dependency_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                    logger=logger, log_every=args.log_every, fade_blur_schedule=fade_blur_schedule)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

                if dist.get_rank() == 0:
                    vq_loss.module.wandb_tracker.log({
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": avg_loss},
                        step=train_steps
                    )
                    # show images and recon images
                    if train_steps % args.vis_every == 0:
                        with torch.no_grad():
                            recons_with_scale = vq_model.module.img_to_reconstructed_img(imgs[:4], last_one=False)
                        image = torch.cat(recons_with_scale + [imgs[:4]], dim=0)
                        image = torch.clamp(image, min=-1, max=1)
                        image = make_grid((image + 1) / 2, nrow=4, padding=0, pad_value=1.0)
                        image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                        image = Image.fromarray(image.astype(np.uint8))

                        vq_loss.module.wandb_tracker.log({"recon_images": [wandb.Image(image)]}, step=train_steps)

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if args.save_best:
                    vq_model.eval()
                    total = 0
                    samples = []
                    gt = []
                    for x, _ in tqdm(val_loader, desc=f'evaluation for step {train_steps:07d}', disable=not rank == 0):
                        with torch.no_grad():
                            x = x.to(device, non_blocking=True)
                            sample = vq_model.module.img_to_reconstructed_img(x)
                            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                            x = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                        
                        sample = torch.cat(dist.nn.all_gather(sample), dim=0)
                        x = torch.cat(dist.nn.all_gather(x), dim=0)
                        samples.append(sample.to("cpu", dtype=torch.uint8).numpy())
                        gt.append(x.to("cpu", dtype=torch.uint8).numpy())

                        total += sample.shape[0]
                    vq_model.train()
                    logger.info(f"Ealuate total {total} files.")
                    dist.barrier()

                    if rank == 0:
                        samples = np.concatenate(samples, axis=0)
                        gt = np.concatenate(gt, axis=0)
                        config = tf.ConfigProto(
                            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
                        )
                        config.gpu_options.allow_growth = True

                        evaluator = Evaluator(tf.Session(config=config),batch_size=32)
                        evaluator.warmup()
                        logger.info("computing reference batch activations...")
                        ref_acts = evaluator.read_activations(gt)
                        logger.info("computing/reading reference batch statistics...")
                        ref_stats, _ = evaluator.read_statistics(gt, ref_acts)
                        logger.info("computing sample batch activations...")
                        sample_acts = evaluator.read_activations(samples)
                        logger.info("computing/reading sample batch statistics...")
                        sample_stats, _ = evaluator.read_statistics(samples, sample_acts)
                        FID = sample_stats.frechet_distance(ref_stats)

                        logger.info(f"traing step: {train_steps:07d}, FID {FID:07f}")
                        # eval code, delete prev if not the best
                        if curr_fid == None:
                            curr_fid = [FID, train_steps]
                        elif FID <= curr_fid[0]:
                            # os.remove(f"{cloud_checkpoint_dir}/{curr_fid[1]:07d}.pt")
                            curr_fid = [FID, train_steps]

                        vq_loss.module.wandb_tracker.log({"eval FID": FID}, step=train_steps)

                    dist.barrier()

                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    # torch.save(checkpoint, cloud_checkpoint_path)
                    # logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

                    if args.save_best:
                        last_checkpoint_path = f"{args.cloud_save_path}/last_ckpt.pt"
                        if os.path.exists(last_checkpoint_path):
                            os.remove(last_checkpoint_path)
                        else:
                            os.makedirs(f"{args.cloud_save_path}", exist_ok=True)
                        torch.save(checkpoint, last_checkpoint_path)
                        logger.info(f"Saved checkpoint in cloud to {last_checkpoint_path}")
                        if curr_fid[1] == train_steps:
                            best_checkpoint_path = f"{args.cloud_save_path}/best_ckpt.pt"
                            torch.save(checkpoint, best_checkpoint_path)
                            logger.info(f"Saved checkpoint in cloud to {best_checkpoint_path}")

                dist.barrier()

        if vqvae_lr_scheduler is not None:
            vqvae_lr_scheduler.step(epoch + 1)
        if disc_lr_scheduler is not None and epoch >= args.disc_epoch_start:
            disc_lr_scheduler.step(epoch + 1 - args.disc_epoch_start)

    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    args = parse_args()
    main(args)
