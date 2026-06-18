#!/bin/bash

export PYTORCH_ENABLE_FUNC_IMPL=1 && \
export PYTORCH_DDP_NO_REBUILD_BUCKETS=1 && \
export TORCH_NCCL_IB_TIMEOUT=23 && \
export NCCL_TIMEOUT=3600 && \
export SETUPTOOLS_USE_DISTUTILS=local && \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \


# Plain flow matching (diagonal velocity regression only — no flow-map / LSD term).
# Same network as the lsd run; differs only in the loss objective.
torchrun --standalone --nproc_per_node=4 train_mf.py \
    --outdir=logs/flow_matching/afhq256/ \
    --data=./data/afhq-256x256.zip \
    --cond=0 --arch=ddpmpp --lr 2e-4 --batch 32 \
    --loss_type=flow_matching \
    --noise_dist=uniform \
    --log_weights=1 \
    --duration=100 \
    --metrics=none \
    # --split=train --train-ratio=0.9 \
    # --resume <path-to-snapshot.pkl>
