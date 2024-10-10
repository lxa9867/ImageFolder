## ImageFolder🚀: Autoregressive Image Generation with Folded Tokens

<div align="center">

[![project page](https://img.shields.io/badge/ImageFolder%20project%20page-lightblue)](https://lxa9867.github.io/works/imagefolder/index.html)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.01756-b31b1b.svg)](https://arxiv.org/abs/2410.01756)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](placeholder)&nbsp;

</div>
<!-- <p align="center" style="font-size: larger;">
  <a href="placeholder">🔥ImageFolder: Autoregressive Image Generation with Folded Tokens</a>
</p> -->

<p align="center">

<div align=center>
	<img src=assets/teaser.png/>
</div>


## Updates 
- (2024.10.03) We are working on advanced training of ImageFolder tokenizer. The code and weights will be released after we finish advanced training.
- (2024.10.01) Repo created. Code and checkpoints will be released soon.

## Ablation (updating)
| ID  | Method                                              | Length | rFID ↓  | gFID ↓ | ACC ↑|
| --- | --------------------------------------------------- | ------ | ------- | ------- |------- |
| 🔶1   | Multi-scale residual quantization (Tian et al., 2024) | 680    | 1.92    | 7.52 | - |
| 🔶2   | + Quantizer dropout                                  | 680    | 1.71    | 6.03 | - |
| 🔶3   | + Smaller patch size K = 11                          | 265    | 3.24    | 6.56 | - |
| 🔶4   | + Product quantization & Parallel decoding           | 265    | 2.06    | 5.96 | - |
| 🔶5   | + Semantic regularization on all branches            | 265    | 1.97    | 5.21 | - |
| 🔶6   | + Semantic regularization on one branch              | 265    | 1.57    | 3.53 | 40.5 |
| 🔷7   | + Stronger discriminator             | 265    | 1.04    | 2.94 | 50.2|
| 🔷8   | + Equilibrium enhancement    | 265    | 0.80    | 2.67 | 50.8|
| 🔷9   | + Parameter searching    | -    | -    | - | - |
| 🔷10   | + Long training    | -    | -    | - | - |

🔶1-6 are already in the released paper, and after that 🔷7+ are advanced training settings used similar to VAR (gFID 3.30).


## Generation

<div align=center>
	<img src=assets/visualization.png/>
</div>

## Visualization of Decomposed Token

<div align=center>
	<img src=assets/token-vis.png/>
</div>

## Acknowledge
We would like to thank the following repositories: [LlamaGen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR) and [ControlVAR](https://github.com/lxa9867/ControlVAR).
## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using
```
@misc{li2024imagefolderautoregressiveimagegeneration,
      title={ImageFolder: Autoregressive Image Generation with Folded Tokens}, 
      author={Xiang Li and Hao Chen and Kai Qiu and Jason Kuen and Jiuxiang Gu and Bhiksha Raj and Zhe Lin},
      year={2024},
      eprint={2410.01756},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.01756}, 
}
```
