<h1 align="center">QuatRoPE</h1>

<p align="center">
    <a href='https://arxiv.org/abs/'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://fz-zsl.github.io/quatrope'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/fzzsl/QuatRoPE/tree/main'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-yellow?style=plastic&logo=huggingface&logoColor=yellow' alt='Checkpoints'>
    </a>
</p>


This repository contains the official PyTorch implementation for the paper:

> Shengli Zhou, Minghang Zheng, Feng Zheng, and Yang Liu. 2026. Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

## Overview

Spatial reasoning focuses on locating target objects based on spatial relations in 3D scenes, which plays a crucial role in developing intelligent embodied agents. Due to the limited availability of 3D scene-language paired data, it is challenging to train models with strong reasoning ability from scratch. Previous approaches have attempted to inject 3D scene representations into the input space of Large Language Models (LLMs) and leverage the pretrained comprehension and reasoning abilities for spatial reasoning. However, models encoding absolute positions struggle to extract spatial relations from prematurely fused features, while methods explicitly encoding all spatial relations (which is quadratic in the number of objects) as input tokens suffer from poor scalability. To address these limitations, we propose **QuatRoPE**, a novel positional embedding method with an input length that is linear to the number of objects, and explicitly calculates pairwise spatial relations through the dot product in attention layers. QuatRoPE's holistic vector encoding of 3D coordinates guarantees a high degree of spatial consistency, maintaining fidelity to the scene’s geometric integrity. Additionally, we introduce the Isolated Gated RoPE Extension (**IGRE**), which effectively limits QuatRoPE's influence to object-related tokens, thereby minimizing interference with the LLM’s existing positional embeddings and maintaining the LLM’s original capabilities. Extensive experiments demonstrate the effectiveness of our approaches.

## Preparation

### Installation

1. Prepare the environment:

```sh
conda create -n quatrope python=3.9.17
conda activate quatrope
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
conda install -c conda-forge openjdk
```

2. Download and install [flash attention](https://github.com/Dao-AILab/flash-attention/releases), we use v2.8.3 in our environment.
3. Download LLM backbone from [Hugging Face](https://huggingface.co/lmsys/vicuna-7b-v1.5), and modify `llama_model_path` in the bash scripts in the `scripts/` directory.
4. Download annotations from [Hugging Face](https://huggingface.co/datasets/ZzZZCHS/Chat-Scene/tree/main/annotations) AND [Yandex Disk](https://disk.yandex.ru/d/LpPJgHg8Qg6BpA) and place them in the `annotations/` directory, more details for data preparation can be found [here](https://github.com/CognitiveAISystems/3DGraphLLM/tree/main/preprocess).
5. Replace `(env_base_path)/lib/python3.9/site-packages/transformers/generation/utils.py` with `generation/utils.py`. `(env_base_path)` can be found by running `conda env list` in the terminal (use the path corresponding to the `quatrope` environment).

### ASR Benchmark

#### Dataset Preparation

The data for the ASR benchmark are in `ASR/scanqa_scanrefer_gt_val.json`, move it under the `annotations/` directory.

#### Zero-Shot Evaluation

1. Modify `calc_scanrefer_score` to `calc_scanrefer_score_asr` in the `evaluate` function of `tasks/train.py`.
2. Run the following script:

```sh
sh scripts/run_gt_eval_asr.sh
```

## Training & Inference

1. Pretrain on ground-truth (GT) segmentation:

```sh
sh scripts/run_gt_pretrain.sh
```

2. Finetune on Mask3D segmentation:

```sh
sh scripts/run_mask3d_finetune.sh
```

3. Evaluation:

```sh
sh scripts/run_mask3d_eval.sh
```

## Models

|  |  | ScanRefer |  | Multi3DRefer |  | SQA3D |
| :-----------------------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| Model                     | Weights      | Acc.@0.25 | Acc.@0.5 | F1@0.25 | F1@0.5 | EM@1 |
| Chat-Scene + QuatRoPE | [Hugging Face](https://huggingface.co/fzzsl/QuatRoPE/blob/main/quatrope-igre-m3d-finetune-7b-knn-0.pth) / [Model Scope](https://www.modelscope.cn/models/Chestnut622/QuatRoPE/file/view/master/quatrope-igre-m3d-finetune-7b-knn-0.pth) | 57.8 | 52.2 | 59.5 | 54.8 | 54.7 |
| 3DGraphLLM + QuatRoPE | [Hugging Face](https://huggingface.co/fzzsl/QuatRoPE/blob/main/quatrope-igre-m3d-finetune-7b-knn-2.pth) / [Model Scope](https://www.modelscope.cn/models/Chestnut622/QuatRoPE/file/view/master/quatrope-igre-m3d-finetune-7b-knn-2.pth) | 58.2 | 52.5 | 60.6 | 56.0 | 55.2 |

Note: The checkpoints are based on Vicuna-7B-v1.5, trained by GT segmentation, and fine-tuned using the Mask3D segmentation.

## Acknowledgement

We would like to thank the open-source code base of [3DGraphLLM](https://github.com/CognitiveAISystems/3DGraphLLM) and the anonymous reviewers for their constructive feedback.

## Citation

If you find this project useful in your research, please consider citing:

```bib

```
