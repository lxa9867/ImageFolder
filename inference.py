################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import dist
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from utils import arg_util, misc
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import build_vae_var
from utils.data import build_dataset
from torch.utils.data import DataLoader
from utils.data_sampler import EvalDistributedSampler
from PIL import Image
from tqdm import tqdm
import wandb
from evaluator import Evaluator
import tensorflow.compat.v1 as tf

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



def main(args):
    MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}
    if dist.get_rank() == 0:
        wandb_tracker = wandb.init(project='VAR_vis', name='debug')
    # /sensei-fs/users/xiangl/exp59-rere_var_d16_200epc/ar-ckpt-best.pth'
    ckpt = torch.load(args.infer_ckpt, map_location='cpu')
    var_ckpt = ckpt['trainer']['var_wo_ddp']
    vae_ckpt = ckpt['trainer']['vae_local']
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        device=args.device, patch_nums=args.patch_nums,
        num_classes=1000, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini, args=args,
    )
    #
    # # load checkpoints
    vae.load_state_dict(vae_ckpt, strict=True)
    var.load_state_dict(var_ckpt, strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    # set args
    seed = 0  # @param {type:"number"}
    torch.manual_seed(seed)
    cfg = args.cfg  # @param {type:"slider", min:1, max:10, step:0.1}
    top_k = args.top_k
    more_smooth = False  # True for more smooth output
    joint_sample = args.joint_sample
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # sample
    print(f'[build PT data] ...\n')
    _, _, dataset_val = build_dataset(
        args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
    )        
    ld_val = DataLoader(
        dataset_val, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size, sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    del dataset_val
    global_batch_size = args.bs
    total = 0
    samples = []
    for step, (x, label) in enumerate(tqdm(ld_val, disable=(dist.get_rank() != 0))):
        label = label.to(args.device, non_blocking=True, dtype=torch.long)
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16,
                                cache_enabled=True):  # using bfloat16 can be faster
                gen_B3HW = var.autoregressive_infer_cfg(B=label.shape[0], label_B=label, cfg=cfg, top_k=top_k, top_p=args.top_p, g_seed=total + dist.get_rank(),
                                                      more_smooth=more_smooth, joint_sample=joint_sample)
                if dist.get_rank() == 0 and step % 10 == 0:
                    chw = torchvision.utils.make_grid(gen_B3HW, nrow=8, padding=0, pad_value=1.0)
                    wandb_tracker.log({"recon_images": [wandb.Image(chw)]},)
                sample = torch.clamp(255 * gen_B3HW, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).contiguous()
                sample = dist.allgather(sample)

            samples.append(sample.to("cpu", dtype=torch.uint8).numpy())
            total += sample.shape[0]

    dist.barrier()
    if dist.get_rank() == 0:
        samples = np.concatenate(samples, axis=0)
        np.random.shuffle(samples)
        config = tf.ConfigProto(
                allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        evaluator = Evaluator(tf.Session(config=config))
        evaluator.warmup()
        ref_acts = evaluator.read_activations('VIRTUAL_imagenet256_labeled.npz')
        ref_stats, ref_stats_spatial = evaluator.read_statistics('VIRTUAL_imagenet256_labeled.npz', ref_acts)
        sample_acts = evaluator.read_activations(samples)
        sample_stats, sample_stats_spatial = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)
        IS = evaluator.compute_inception_score(sample_acts[0])
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        print(f"cfg: {args.cfg} topk: {args.top_k} topp: {args.top_p} FID: {FID} IS: {IS} Precision: {prec} Recall: {recall}")
    
    dist.barrier()
    print("Done")
        


if __name__ == '__main__':
    args = arg_util.init_dist_and_get_args()
    print(args.cfg, args.top_k, args.top_p)
    main(args)