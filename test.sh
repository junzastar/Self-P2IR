#!/bin/bash
n_gpu=1 # number of gpu to use
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port 12121  main.py --config="configs/test/main_config.yaml" --gpus=$n_gpu