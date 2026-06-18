#!/bin/bash

export PYTORCH_ENABLE_FUNC_IMPL=1 && \
export PYTORCH_DDP_NO_REBUILD_BUCKETS=1 && \
export TORCH_NCCL_IB_TIMEOUT=23 && \
export NCCL_TIMEOUT=3600 && \
export SETUPTOOLS_USE_DISTUTILS=local && \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \


torchrun --standalone --nproc_per_node=4 train_mf.py \
    --detach_tgt=1 \
    --outdir=logs/aniso_flow_map/flow_matching_model/ \
    --data=./data/afhq-256x256.zip \
    --cond=0 --arch=ddpmpp --lr 2e-4 --batch 32 \
    --noise_dist=uniform \
    --loss_type=lsd \
    --log_weights=1 \
    --duration=100 \
    --data_proportion=0.75 \
    --lsd_warmup_kimg=0 \
    --metrics=none \
    # --batch-gpu 32 \
    # --split=train --train-ratio=0.9 \

    # --resume /home/nvidia/easy_meanflow/logs/aniso_flow_map/lsd_correct_partition/00024-cifar10-32x32-uncond-ddpmpp-mf-gpus4-batch512-fp32/network-snapshot-005018.pkl \

# torchrun --standalone --nproc_per_node=4 train_mf.py \
#     --detach_tgt=1 \  
#     --outdir=logs/aniso_flow_map/lsd_correct_partition/ \
#     --data=./data/celeba/celeba-64x64.zip \
#     --cond=0 --arch=ddpmpp --lr 2e-4 --batch 200 \
#     --anisotropic=1 \
#     --aniso_min_box_size=10 \
#     --aniso_max_box_size=40 \
#     --noise_dist=uniform \
#     --loss_type=lsd \
#     --log_weights=1 \
#     --duration=100 \
#     --data_proportion=0.75 \
    # --resume /home/william/easy_meanflow/logs/mf/MF00/00002-cifar10-32x32-uncond-ddpmpp-mf-gpus4-batch256-fp32/training-state-200000.pt \


# torchrun --standalone --nproc_per_node=4 train_mf.py \
#     --detach_tgt=1 \
#     --outdir=logs/aniso_flow_map/lsd_correct_partition/ \
#     --data=./data/cifar10-32x32.zip \
#     --cond=0 --arch=ddpmpp --lr 2e-4 --batch 512 \
#     --anisotropic=1 \
#     --aniso_min_box_size=5 \
#     --aniso_max_box_size=23 \
#     --noise_dist=uniform \
#     --loss_type=lsd \
#     --log_weights=1 \
#     --duration=100 \
#     --data_proportion=0.75 \
#     # --resume /home/william/easy_meanflow/logs/mf/MF00/00002-cifar10-32x32-uncond-ddpmpp-mf-gpus4-batch256-fp32/training-state-200000.pt \
