#!/bin/bash
n_gpu=3 # number of gpu to use
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=$n_gpu  main.py --config="configs/train/main_config.yaml" --gpus=$n_gpu