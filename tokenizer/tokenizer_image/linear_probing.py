import os
import sys
sys.path.append('/home/xiangl/LlamaGen')
import logging
import json
import numpy as np
import torch.distributed
from tqdm.auto import tqdm
from PIL import Image
from logging import getLogger as get_logger
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import DataLoader, default_collate
from utils2 import init_distributed_device, is_global_primary, is_primary, seed_everything, str2bool
from tokenizer.tokenizer_image.msvq_model import VQ_models
from datasets import create_dataset, fast_collate, PrefetchLoader, Normalize, Denormalize

from timm.optim import create_optimizer_v2 as create_optimizer
from timm.scheduler import create_scheduler_v2 as create_scheduler

import argparse

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str,
                        default='configs/vqgan/imagenet/vqvae_vq_dinov2base_v4096z16n64_pretrained_ae.yaml',
                        help="config file used to specify parameters")

    # data
    parser.add_argument("--data_dir", type=str, default='imagenet/train', help="data folder")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="dataset name")
    parser.add_argument("--val_data_dir", type=str, default='imagenet/val', help="data folder")
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--num_classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument("--use_prefetcher", type=str2bool, default=True, help="use prefetch")

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--min_lr", type=float, default=5e-5, help="end learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help='lr scheduler')
    parser.add_argument("--lr_warmup_epochs", type=float, default=1, help="warmup epochs")
    parser.add_argument("--log_interval", type=int, default=50, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1000, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=int, default=1, help='save interval')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')
    parser.add_argument("--gradient_clip", type=float, default=1.0, help='gradient clip')
    parser.add_argument("--torchcompile", type=str2bool, default=False, help='use torch compile')
    parser.add_argument("--report_to", type=str, default='wandb', help='report to',
                        choices=['wandb', 'tensorboard', 'none'])
    parser.add_argument("--resume", type=str, default=None, help='resume from pre-trained checkpoint')
    parser.add_argument("--auto_resume", type=str2bool, default=False, help='auto resume from latest checkpoint')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                        help='mixed precision')
    parser.add_argument("--ema", type=float, default=0, help='ema updates of the models')
    parser.add_argument("--beta1", type=int, default=0.9, help='beta1 for adam')
    parser.add_argument("--beta2", type=int, default=0.99, help='beta2 for adam')
    parser.add_argument("--quantizer_lr_multiplier", type=float, default=1.0, help='lr multiplier for quantization')
    parser.add_argument("--compile", type=str2bool, default=False, help='use torch compile')

    # loss weight
    parser.add_argument("--disc_adaptive", type=str2bool, default=True,
                        help="flag of whether to use adaptive discriminator weight")
    parser.add_argument("--disc_loss_start", type=float, default=0,
                        help="starting threshold of adaptive discriminator weight for discriminator training")
    parser.add_argument("--disc_loss_weight", type=float, default=0.8, help="discriminator loss weight")
    parser.add_argument("--gen_disc_loss_weight", type=float, default=0.1,
                        help="discriminator loss weight of generator")
    parser.add_argument("--gen_disc_loss_type", type=str, default='non-saturating',
                        choices=["hinge", "vanilla", "non-saturating"], help='generator loss type')
    parser.add_argument("--disc_loss_type", type=str, default='hinge', choices=["hinge", "vanilla", "non-saturating"],
                        help='discriminator loss type')
    parser.add_argument("--disc_model", type=str, default='patchgan', choices=["patchgan", "stylegan"],
                        help='discriminator loss type')
    parser.add_argument("--lecam_loss_weight", type=float, default=0.0,
                        help="lecam regularization loss weight of discriminator")
    parser.add_argument("--codebook_loss_weight", type=float, default=1.0, help="codebook loss weight")
    parser.add_argument("--perceptual_loss_weight", type=float, default=0.1, help="perceptual loss weight")
    parser.add_argument("--logit_scale_loss_weight", type=float, default=0.1, help="logit_scale loss weight")
    parser.add_argument("--rec_loss_weight", type=float, default=1.0, help="rec loss weight")

    parser.add_argument("--ent_loss_weight", type=float, default=0.1, help="entropy loss weight")
    parser.add_argument("--ent_loss_weight_end", type=float, default=0.0, help="entropy loss weight")
    parser.add_argument("--ent_loss_start", type=float, default=1.0, help="start to add entropy loss")
    parser.add_argument("--ent_loss_annealing_steps", type=float, default=2000,
                        help="steps to anneal entropy loss weight")
    parser.add_argument("--sem_loss_weight", type=float, default=0.01, help="semantic loss weight")
    parser.add_argument("--ent_sample_min_loss_weight", type=float, default=1.0,
                        help="sample entropy minimization loss weight")
    parser.add_argument("--ent_batch_max_loss_weight", type=float, default=1.0,
                        help="batch entropy maximization loss weight")

    # vqvae
    parser.add_argument("--recon_loss", type=str, default='l1', choices=['l1', 'l2'], help='reconstruction loss')
    parser.add_argument("--quantizer", type=str, default='vq',
                        choices=['vq', 'gumbel_vq', 'st_gumbel_vq', 'ema_vq', 'oc_vq', 'diff_vq', 'diff_vq2',
                                 'diff_vq_fix'], help='quantizer type')
    parser.add_argument("--encoder", type=str, default='dinov2', help='encoder model type')
    parser.add_argument("--decoder", type=str, default='dinov2', help='deocder model type')
    parser.add_argument("--encoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='encoder model name')
    parser.add_argument("--encoder_model_pretrained", type=str2bool, default=True,
                        help='encoder model load pretrained checkpoint')
    parser.add_argument("--encoder_patch_size", type=int, default=16, help='encoder patch size')
    parser.add_argument("--encoder_tuning", type=str, default='lora', help='encoder tuning method')
    parser.add_argument("--encoder_tuning_lora_r", default=8, type=int, help='encoder tuning lora r')
    parser.add_argument("--encoder_drop_path", type=float, default=0.0, help='encoder droppath rate')
    parser.add_argument("--decoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='deocder model name')
    parser.add_argument("--decoder_model_pretrained", type=str2bool, default=True,
                        help='decoder model load pretrained checkpoint')
    parser.add_argument("--decoder_patch_size", type=int, default=16, help='decoder patch size')
    parser.add_argument("--decoder_drop_path", type=float, default=0.0, help='decoder droppath rate')
    parser.add_argument("--decoder_to_pixel", type=str, default='linear', help='decoder to pixel',
                        choices=['linear', 'conv', 'ada_conv', 'siren'])
    parser.add_argument("--decoder_use_rope", type=str2bool, default=False, help='decoder use RoPE')
    parser.add_argument("--decoder_cond_latent", type=str2bool, default=False,
                        help='use dino latent to initialize latent tokens (mask token)')
    parser.add_argument("--decoder_tuning", type=str, default='lora', help='deocder tuning method')
    parser.add_argument("--decoder_tuning_lora_r", default=8, type=int, help='decoder tuning lora r')
    parser.add_argument("--pretrained_path", type=str, default=None, help='pretrained model path')
    parser.add_argument("--semantic_guide", type=str, default='none', help='semantic guidance on latent tokens')
    parser.add_argument("--sem_loss_scale", type=float, default=15.0, help='scale for clip loss')
    parser.add_argument("--renorm_input", type=str2bool, default=False, help='normalize input images')

    parser.add_argument("--vocab_size", type=int, default=4096, nargs='+', help="codebook size")
    parser.add_argument("--z_channels", type=int, default=32, help="latent size of vqvae")
    parser.add_argument("--num_latent_tokens", type=int, default=32, help="number of latent tokens")
    parser.add_argument("--codebook_norm", type=str2bool, default=True, help='normalize codebook')
    parser.add_argument("--use_gumbel", type=str2bool, default=False, help='use gumbel softmax for probs')
    parser.add_argument("--commit_loss_weight", type=float, default=0.0, help="commit loss weight")
    parser.add_argument("--kl_loss_weight", type=float, default=5e-4, help="kl loss weight")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="ema decay for embeddings of ema quantizer")
    parser.add_argument("--oc_anchor", type=str, default='cloest', help="online cluster anchor",
                        choices=['closest', 'random', 'projrandom'])
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0, help="contrastive loss weight")
    parser.add_argument("--freq_loss_weight", type=float, default=0.0, help="freq loss weight")
    parser.add_argument("--disc_r1_gamma", type=float, default=0.0, help="disc do r1 reg")
    parser.add_argument("--use_diffaug", type=str2bool, default=False, help='use diff aug')
    parser.add_argument("--init_logit_scale", type=float, default=10, help="initial logit scale before log")
    parser.add_argument("--max_logit_scale", type=float, default=200, help="maximum logit scale before log")

    parser.add_argument("--v_patch_nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], nargs='+',
                        help="number of patch numbers of each scale")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--output_path", type=str, default='output/linear_probing', help='output model path')
    parser.add_argument("--enc_type", type=str, default="cnn")
    parser.add_argument("--dec_type", type=str, default="cnn")
    # fFirst parse of command-line args to check for config file
    # args = parser.parse_args()

    # # If a config file is specified, load it and set defaults
    # if args.config is not None:
    #     with open(args.config, 'r', encoding='utf-8') as f:
    #         file_yaml = yaml.YAML()
    #         config_args = file_yaml.load(f)
    #         parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.bn(x)
        out = self.linear(x)
        return out


@torch.no_grad()
def extract_feature(vqvae, images, args):

    if args.distributed:

        z_e = vqvae.module.encoder(images)
        if args.enc_type == 'dinov2':
            b, l, c = z_e.shape
            z_e = z_e.view(b, 16, 16, c)
            z_e = z_e.permute(0, 3, 1, 2)
        z_e = vqvae.module.quant_conv(z_e)
        # z_q, _, _ = vqvae.module.quantize(z_e)

    else:
        z_e = vqvae.encoder(images)
        if args.enc_type == 'dinov2':
            b, l, c = z_e.shape
            z_e = z_e.view(b, 16, 16, c)
            z_e = z_e.permute(0, 3, 1, 2)
        z_e = vqvae.quant_conv(z_e)
        # z_q, _, _ = vqvae.quantize(z_e)

    return z_e


def train_epoch(vqvae, linear_classifier, train_dataloader, optimizer, device, scaler, args):
    criterion = torch.nn.CrossEntropyLoss()
    linear_classifier.train()
    train_dtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    total_loss = 0
    total_correct = 0
    total_samples = 0
    if args.renorm_input:
        denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=device)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)

    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not is_primary(args)):
        # features, labels = batch
        # features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        images, labels = batch
        if not args.use_prefetcher:
            images = images.to(device)
            labels = labels.to(device)

        if args.renorm_input:
            input_images = denormalize(images)
            input_images = normalize(input_images)
        else:
            input_images = images

        with torch.cuda.amp.autocast(dtype=train_dtype):

            features = extract_feature(vqvae, input_images, args).detach()

            features = features.flatten(2).permute(0, 2, 1)
            logits = linear_classifier(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

        if is_primary(args) and idx % 25 == 0:
            logger.info(f"Training Loss: {loss.item():.4f}")
            logger.info(f"Training Acc: {total_correct / total_samples * 100.0:.4f}")
    return total_loss / len(train_dataloader), total_correct / total_samples * 100.0


def evaluate(vqvae, linear_classifier, val_dataloader, device, args):
    dtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    criterion = torch.nn.CrossEntropyLoss()
    linear_classifier.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    if args.renorm_input:
        denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=device)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)

    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader), disable=not is_primary(args)):
            # features, labels = batch
            # features, labels = features.to(device), labels.to(device)
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            if args.renorm_input:
                input_images = denormalize(images)
                input_images = normalize(input_images)
            else:
                input_images = images


            with torch.cuda.amp.autocast(dtype=dtype):
                features = extract_feature(vqvae, input_images, args)
                features = features.flatten(2).permute(0, 2, 1)
                # z = extract_feature(vqvae, images, args)
                logits = linear_classifier(features)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(val_dataloader), total_correct / total_samples * 100.0



def main():

    args = parse_args()

    # seed
    seed_everything(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
        os.environ['HF_HOME'] = f'./hf_cache_{args.rank}/'
        os.environ['TRANSFORMERS_CACHE'] = f'./hf_cache_{args.rank}/'
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # create and load model
    logger.info("Creating model")
    # create and load model
    vqvae = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        v_patch_nums=args.v_patch_nums,
        enc_type=args.enc_type,
        dec_type=args.dec_type,
        semantic_guide=args.semantic_guide,
    )
    vqvae.to(device)
    vqvae.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    vqvae.load_state_dict(model_weight)
    del checkpoint

    # create linear classifier
    linear_classifier = LinearClassifier(vqvae.codebook_embed_dim, args.num_classes)
    linear_classifier = linear_classifier.to(device)

    if args.distributed:
        if is_primary(args):
            logger.info("Using native Torch DistributedDataParallel.")
        vqvae = NativeDDP(vqvae, device_ids=[device], find_unused_parameters=True)
        linear_classifier = NativeDDP(linear_classifier, device_ids=[device], find_unused_parameters=True)

    logger.info("Creating dataset")
    train_dataset = create_dataset(args.dataset_name, args.data_dir, args.image_size, is_train=True, use_prefetcher=args.use_prefetcher)
    valid_dataset = create_dataset(args.dataset_name, args.val_data_dir, args.image_size, is_train=False, use_prefetcher=False)
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    shuffle = sampler is None
    collate_fn = fast_collate if args.use_prefetcher else default_collate
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)
    if args.use_prefetcher:
        train_dataloader = PrefetchLoader(train_dataloader, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=device)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    total_batch_size = args.batch_size * args.world_size

    # create output folder
    # output_dir = os.path.join(args.output_dir, args.run_name, 'evaluations')
    output_dir = args.output_path
    output_dir = os.path.join(output_dir, 'evaluations')
    os.makedirs(output_dir, exist_ok=True)
    lp_model_dir = os.path.join(output_dir, args.dataset_name)
    os.makedirs(lp_model_dir, exist_ok=True)

    optimizer = create_optimizer(linear_classifier, opt=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scheduler, _ = create_scheduler(
        sched='step',
        decay_milestones=[int(args.num_epochs * 0.3), int(args.num_epochs * 0.6), int(args.num_epochs * 0.9)],
        optimizer=optimizer,
        patience_epochs=0,
        step_on_epochs=True,
        num_epochs=args.num_epochs,
        warmup_epochs=args.lr_warmup_epochs,
        min_lr=1e-6,
    )


    # train linear classifier
    logger.info("Start training linear classifier")
    max_accuracy = 0
    for epoch in range(args.num_epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        train_loss, train_acc = train_epoch(vqvae, linear_classifier, train_dataloader, optimizer, device, scaler, args)
        val_loss, val_acc = evaluate(vqvae, linear_classifier, val_dataloader, device, args)

        if is_global_primary(args):
            if args.distributed:
                torch.save(linear_classifier.module.state_dict(), os.path.join(lp_model_dir, f'epoch_{epoch}.pth'))
            else:
                torch.save(linear_classifier.state_dict(), os.path.join(lp_model_dir, f'epoch_{epoch}.pth'))

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logger.info(f"Saving best model with accuracy {max_accuracy}")
            if args.distributed:
                torch.save(linear_classifier.module.state_dict(), os.path.join(lp_model_dir, 'best.pth'))
            else:
                torch.save(linear_classifier.state_dict(), os.path.join(lp_model_dir, 'best.pth'))

        if is_primary(args):
            logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc}, val_loss={val_loss}, val_acc={val_acc}")
            logger.info(f"Best accuracy so far: {max_accuracy}")

        scheduler.step(epoch + 1)
    results = {
            'best_lp_accuracy': max_accuracy
    }

    # logger.info("Start training k-nn")

    if is_primary(args):
        logger.info("Finished training")
        logger.info(f"Best accuracy: {max_accuracy}")


        with open(os.path.join(output_dir, 'linear_results.json'), 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()

