# Robust Latent Matters: Boosting Image Generation with Sampling Error Synthesis

<div align="center">

[![paper](https://img.shields.io/badge/arXiv%20paper-2503.08354-b31b1b.svg)](https://arxiv.org/pdf/2503.08354)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)]()&nbsp;

</div>

<p>
<img src="assets/robust-teaser.png" alt="teaser" width=90% height=90%>
</p>

## Model Zoo
| Model | Link | FID |
| ------------- | ------------- | ------------- |
| RAR-B | [checkpoint](https://huggingface.co/qiuk6/RobustTok/resolve/main/rar-b.bin?download=true)| 1.83 (generation) |
| RAR-L | [checkpoint](https://huggingface.co/qiuk6/RobustTok/resolve/main/rar-l.bin?download=true)| 1.60 (generation) |

## Inference Code



## Training Code
```
accelerate launch scripts/train_rar.py experiment.project="rar" experiment.name="rar_b" experiment.output_dir="rar_b" model.generator.hidden_size=768 model.generator.num_hidden_layers=24 model.generator.num_attention_heads=16 model.generator.intermediate_size=3072 config=configs/generator/robustTok-rar.yaml dataset.params.pretokenization=/path/to/pretokenized.jsonl mode.vq_ckpt=/path/to/RobustTok.pt

accelerate launch scripts/train_rar.py experiment.project="rar" experiment.name="rar_l" experiment.output_dir="rar_l" model.generator.hidden_size=1024 model.generator.num_hidden_layers=24 model.generator.num_attention_heads=16 model.generator.intermediate_size=4096 config=configs/generator/robustTok-rar.yaml dataset.params.pretokenization=/path/to/pretokenized.jsonl mode.vq_ckpt=/path/to/RobustTok.pt
```

## Visualization Result

<p>
<img src="assets/robust-vis.png" alt="robust-vis" width=95% height=95%>
</p>