#!/usr/bin/env bash
# X-Fi-JEPA comparison suite (JEPAREADME §5.4): runs groups A/B/B'/C with the SAME
# evaluation protocol (7 modality subsets, sample granularity, worst-subset).
# Run from XRF55_HAR/. Shared 3xV100 server: check `nvidia-smi` first and set GPU_ID
# to a free card; every command below uses exactly one GPU.

set -e

GPU_ID=0                                                # <-- pick a FREE gpu (nvidia-smi)
DATASET=../data/XRF55_Dataset_split                     # symlink split tree (make_xrf55_split.py)
JEPA_CKPT=./jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/epoch_100.pth  # JEPA pretrain output
XFI_CKPT=./pre-trained_weights/XRF55_har_checkpoint.pt  # group A = official released model (2026-09-19)
# Self-trained alternative for group A (after running §1.3 run.py):
#   XFI_CKPT=./pre-trained_weights/checkpoint_<时间戳>.pth

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "==== group A: X-Fi supervised (eval only) ===="
conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method xfi_supervised --dataset "$DATASET" --xfi-weights "$XFI_CKPT"

echo "==== group B: X-Fi-JEPA pretrain + linear probe ===="
conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_linear_probe --dataset "$DATASET" --pretrained "$JEPA_CKPT"

echo "==== group B': X-Fi-JEPA pretrain + finetune (optional, recommended) ===="
conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_finetune --dataset "$DATASET" --pretrained "$JEPA_CKPT"

echo "==== group C: random-init lower bound + linear probe ===="
conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method random_init --dataset "$DATASET"

echo "==== all done; per-method JSON results are in jepa_eval_results/ ===="
