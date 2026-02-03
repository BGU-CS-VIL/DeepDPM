#!/bin/bash
# Run MNIST Imbalanced experiment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$REPO_ROOT"

python reimplementation/scripts/train_from_config.py \
    --config reimplementation/configs/experiments/mnist_imbalanced.yaml \
    --embeddings pretrained_embeddings/umap_embedded_datasets/MNIST_IMBALANCED/train_data.pt \
    --labels pretrained_embeddings/umap_embedded_datasets/MNIST_IMBALANCED/train_labels.pt \
    --name "MNIST Imbalanced"
