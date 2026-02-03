#!/bin/bash
# Run USPS experiment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$REPO_ROOT"

python reimplementation/scripts/train_from_config.py \
    --config reimplementation/configs/experiments/usps.yaml \
    --embeddings pretrained_embeddings/umap_embedded_datasets/USPS/train_data.pt \
    --labels pretrained_embeddings/umap_embedded_datasets/USPS/train_labels.pt \
    --name "USPS"
