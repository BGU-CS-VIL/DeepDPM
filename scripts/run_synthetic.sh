#!/bin/bash
# Run Synthetic GMM experiment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$REPO_ROOT"

# Note: Synthetic experiment uses generated data, not pre-saved embeddings
python reimplementation/scripts/train_synthetic.py
