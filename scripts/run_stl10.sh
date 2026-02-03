#!/bin/bash
# Run STL10 experiment (MOCO embeddings)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$REPO_ROOT"

python reimplementation/scripts/train_from_config.py \
    --config reimplementation/configs/experiments/stl10.yaml \
    --embeddings pretrained_embeddings/MOCO/STL10/train_data.pt \
    --labels pretrained_embeddings/MOCO/STL10/train_labels.pt \
    --name "STL10 (MOCO embeddings)"
