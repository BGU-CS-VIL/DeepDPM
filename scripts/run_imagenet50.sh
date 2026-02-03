#!/bin/bash
# Run ImageNet-50 experiment (MOCO embeddings, 128D)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$REPO_ROOT"

python reimplementation/scripts/train_from_config.py \
    --config reimplementation/configs/experiments/imagenet50.yaml \
    --embeddings pretrained_embeddings/MOCO/IMAGENET_50/train_data.pt \
    --labels pretrained_embeddings/MOCO/IMAGENET_50/train_labels.pt \
    --name "ImageNet-50 (MOCO embeddings)"
