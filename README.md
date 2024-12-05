## XQ-GAN🚀: An Open-source Image Tokenization Framework for Autoregressive Generation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.01756-b31b1b.svg)](https://arxiv.org/abs/2412.01762)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/qiuk6/XQ-GAN/tree/main)&nbsp;

</div>
<!-- <p align="center" style="font-size: larger;">
  <a href="placeholder">🔥ImageFolder: Autoregressive Image Generation with Folded Tokens</a>
</p> -->

<p align="center">

<div align=center>
	<img src=assets/xqgan.png/>
</div>

## ImageFolder🚀: Autoregressive Image Generation with Folded Tokens

<div align="center">

[![project page](https://img.shields.io/badge/ImageFolder%20project%20page-lightblue)](https://lxa9867.github.io/works/imagefolder/index.html)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.01756-b31b1b.svg)](https://arxiv.org/abs/2410.01756)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/ang9867/imagefolder/tree/main)&nbsp;
</div>

<p align="center">

<div align=center>
	<img src=assets/teaser.png/>
</div>


## Updates
- (2024.12.03) LFQ and BSQ tokenizer supported. BSQ-style VAR and REPA in autoregressive training will be supported shortly.
- (2024.12.03) XQ-GAN initial code released. ImageFolder is compatible in XQ-GAN.
- (2024.12.02) ImageFolder's code has been released officially at [Adobe Research Repo](https://github.com/adobe-research/ImageFolder).

## Features

<p align="center">

<div align=center>
	<img src=assets/table.png/>
</div>

XQ-GAN is a highly flexible framework that supports the combination of several advanced quantization approaches, backbone architectures, and training recipes (semantic alignment, discriminators, and auxiliary losses). In addition, we also provide finetuning support with full, LORA, and frozen from pre-trained weights.

<p align="center">

<div align=center>
	<img src=assets/quantizer.png  width="500" />
</div>

We implemented a hierarchical quantization approach, which first decides the product quantization (PQ) and then the residual quantization (RQ). The minimum unit of this design consists of vector quantization (VQ), lookup-free quantization (LFQ), and binary spherical quantization (BSQ). A vanilla VQ can be achieved in this framework by setting the product branch and residual depth to 1.

## Model Zoo

We provide pre-trained tokenizers for image reconstruction on ImageNet, LAION-400M (natural image), and IMed361M (multimodal medical image) 256x256 resolution.

<p align="center">

<div align=center>
	<img src=assets/data.png/>
</div>

| Training | Type | Codebook | Latent res. | rFID |                                 Link                                  | Config |
| :------: | :--: | :-----------: | :---------: | :----: | :-------------------------------------------------------------------: | :----: |
| ImageNet | VP2  |     4096      |   16x16    |  0.90  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/vq-4096/best_ckpt.pt?download=true)  | VP2-4096.yaml |
| ImageNet | VP2  |     16384     |   16x16    |  0.64  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/vq-16384/best_ckpt.pt?download=true) | VP2-16384.yaml |

| Training |   Type   | Codebook | Latent res. | rFID |  Link  | Config |
| :------: | :------: | :-----------: | :---------: | :----: | :----: | :----: |
| ImageNet | MSBR10P2 |     4096      | 1x1->11x11  |  0.86  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSBR10P2-4096/best_ckpt.pt?download=true) | MSBR10P2-4096.yaml |
| ImageNet | MSBR10P2 |     16384     | 1x1->11x11  |  0.78  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSBR10P2-16384/best_ckpt.pt?download=true) | MSBR10P2-16384.yaml |

|  Training  |   Type   | Codebook | Latent res. | rFID |                                                  Link                                                  | Config |
| :--------: | :------: | :-----------: | :---------: | :----: | :----------------------------------------------------------------------------------------------------: | :----: |
|  ImageNet  | MSVR10P2 |     4096      | 1x1->11x11  |  0.80  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSVR10P2-4096/best_ckpt.pt?download=true)  | MSVR10P2-4096.yaml |
|  ImageNet  | MSVR10P2 |     8192      | 1x1->11x11  |  0.70  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSVR10P2-8192/best_ckpt.pt?download=true)  | MSVR10P2-8192.yaml |
|  ImageNet  | MSVR10P2 |     16384     | 1x1->11x11  |  0.67  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSVR10P2-16384/best_ckpt.pt?download=true) | MSVR10P2-16384.yaml |
|  IMed  | MSVR10P2 |     4096      | 1x1->11x11  |   -    |    [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/IMed-MSVR10P2-4096/IMed361M.pt?download=true)    | MSVR10P2-4096.yaml |
| LAION | MSVR10P2 |     4096      | 1x1->11x11  |   -    |     [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/LAION-MSVR10P2-4096/laion.pt?download=true) | MSVR10P2-4096.yaml |

---

We provide a pre-trained generator for class-conditioned image generation using MSVR10P2 ([ImgaeFolder's setting](https://arxiv.org/abs/2410.01756)) on ImageNet 256x256 resolution. More support will be made soon.

| Type | Dataset  | Model Size | gFID |                                                   Link                                                    | Resolution |
| :--: | :------: | :--------: | :----: | :-------------------------------------------------------------------------------------------------------: | :--------: |
| VAR  | ImageNet |    362M    |  2.60  | [Huggingface](https://huggingface.co/qiuk6/XQ-GAN/resolve/main/VAR-d17-MSVR10P2-4096/ar-ckpt-last.pth?download=true) |  256x256   |

## Installation

Install all packages as

```
conda env create -f environment.yml
```

## Dataset

We download the ImageNet2012 from the website and collect it as

```
ImageNet2012
├── train
└── val
```

If you want to train or finetune on other datasets, collect them in the format that ImageFolder (pytorch's [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)) can recognize.

```
Dataset
├── train
│   ├── Class1
│   │   ├── 1.png
│   │   └── 2.png
│   ├── Class2
│   │   ├── 1.png
│   │   └── 2.png
├── val
```

## Training code for tokenizer

Please login to Wandb first using

```
wandb login
```

rFID will be automatically evaluated and reported on Wandb. The checkpoint with the best rFID on the val set will be saved. We provide basic configurations in the "configs" folder. 

Warning❗️: You may want to modify the metric to save models as rFID is not closely correlated to gFID. PSNR and SSIM are also good choices.

```
torchrun --nproc_per_node=8 tokenizer/tokenizer_image/xqgan_train.py --config configs/xxxx.yaml
```

Please modify the configuration file as needed for your specific dataset. We list some important ones here.

```
vq_ckpt: ckpt_best.pt                # resume
cloud_save_path: output/exp-xx       # output dir
data_path: ImageNet2012/train        # training set dir
val_data_path: ImageNet2012/val      # val set dir
enc_tuning_method: 'full'            # ['full', 'lora', 'frozen']
dec_tuning_method: 'full'            # ['full', 'lora', 'frozen']
codebook_embed_dim: 32               # codebook dim
codebook_size: 4096                  # codebook size
product_quant: 2                     # PQ branch number
v_patch_nums: [16,]                  # latent resolution for RQ ([16,] is equivalent to vanilla VQ)
codebook_drop: 0.1                   # quantizer dropout rate if RQ is applied
semantic_guide: dinov2               # ['none', 'dinov2', 'clip']
disc_epoch_start: 56	             # epoch that discriminator starts
disc_type: dinodisc		     # discriminator type
disc_adaptive_weight: true	     # adaptive weight for discriminator loss
ema: true                            # use ema to update the model
num_latent_code: 256		     # latent token number (must equals to the v_patch_nums[-1] ** 2）
start_drop: 3			     # quantizer dropout starts depth
```

## Tokenizer linear probing

```
torchrun --nproc_per_node=8 tokenizer/tokenizer_image/linear_probing.py --config configs/msvq.yaml
```

## Training code for VAR (only support MSVRP now)

We follow the VAR training code and our training cmd for reproducibility is

```
torchrun --nproc_per_node=8 train.py --bs=768 --alng=1e-4 --fp16=1 --alng=1e-4 --wpe=0.01 --tblr=8e-5 --data_path /mnt/localssd/ImageNet2012/ --encoder_model vit_base_patch14_dinov2.lvd142m --decoder_model vit_base_patch14_dinov2.lvd142m --product_quant 2 --semantic_guide dinov2 --num_latent_tokens 121 --v_patch_nums 1 1 2 3 3 4 5 6 8 11 --pn 1_1_2_3_3_4_5_6_8_11 --patch_size 11 --vae_ckpt /path/to/ckpt.pt --sem_half True
```

## Inference code for VAR

```
torchrun --nproc_per_node=2 inference.py --infer_ckpt /path/to/ckpt --data_path /path/to/ImageNet --encoder_model vit_base_patch14_dinov2.lvd142m --decoder_model vit_base_patch14_dinov2.lvd142m --product_quant 2 --semantic_guide dinov2 --num_latent_tokens 121 --v_patch_nums 1 1 2 3 3 4 5 6 8 11 --pn 1_1_2_3_3_4_5_6_8_11 --patch_size 11 --sem_half True --cfg 3.25 3.25 --top_k 750 --top_p 0.95
```

## Ablation of MSVR10P2

| ID  | Method                                                | Length | rFID ↓ | gFID ↓ | ACC ↑ |
| --- | ----------------------------------------------------- | ------ | ------ | ------ | ----- |
| 🔶1 | Multi-scale residual quantization (Tian et al., 2024) | 680    | 1.92   | 7.52   | -     |
| 🔶2 | + Quantizer dropout                                   | 680    | 1.71   | 6.03   | -     |
| 🔶3 | + Smaller patch size K = 11                           | 265    | 3.24   | 6.56   | -     |
| 🔶4 | + Product quantization & Parallel decoding            | 265    | 2.06   | 5.96   | -     |
| 🔶5 | + Semantic regularization on all branches             | 265    | 1.97   | 5.21   | -     |
| 🔶6 | + Semantic regularization on one branch               | 265    | 1.57   | 3.53   | 40.5  |
| 🔷7 | + Stronger discriminator                              | 265    | 1.04   | 2.94   | 50.2  |
| 🔷8 | + Equilibrium enhancement                             | 265    | 0.80   | 2.60   | 58.0  |

🔶1-6 are already in the released paper, and after that 🔷7+ are advanced training settings used similar to VAR (gFID 3.30).

## Generation

<div align=center>
	<img src=assets/visualization.png/>
</div>

## Acknowledge

We would like to thank the following repositories: [LlamaGen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR) and [ControlVAR](https://github.com/lxa9867/ControlVAR).

## Citation

If our work assists your research, feel free to give us a star ⭐ or cite us using

```
@article{li2024imagefolder,
  title={Imagefolder: Autoregressive image generation with folded tokens},
  author={Li, Xiang and Qiu, Kai and Chen, Hao and Kuen, Jason and Gu, Jiuxiang and Raj, Bhiksha and Lin, Zhe},
  journal={arXiv preprint arXiv:2410.01756},
  year={2024}
}
```
```
@misc{li2024xqganopensourceimagetokenization,
      title={XQ-GAN: An Open-source Image Tokenization Framework for Autoregressive Generation}, 
      author={Xiang Li and Kai Qiu and Hao Chen and Jason Kuen and Jiuxiang Gu and Jindong Wang and Zhe Lin and Bhiksha Raj},
      year={2024},
      eprint={2412.01762},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.01762}, 
}
```
