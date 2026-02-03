# DeepDPM: Deep Clustering With An Unknown Number of Clusters
This repository contains the official implementation of our CVPR 2022 paper:
> [**DeepDPM: Deep Clustering With An Unknown Number of Clusters**](https://arxiv.org/abs/2203.14309)
>
> [Meitar Ronen](https://www.linkedin.com/in/meitar-ronen/), [Shahaf Finder](https://shahaffind.github.io) and [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/index.htm).

[![arXiv](https://img.shields.io/badge/arXiv-2203.14309-b31b1b.svg?style=flat)](https://arxiv.org/abs/2203.14309)

DeepDPM clustering example on 2D data.<br />
On the left: DeepDPM's predicted clusters' assignments, centers and covariances. On the right: Clusters colored by the GT labels, and the net's decision boundary.
<br>
<p align="center">
<img src="images/clustering_example.gif" width="750" height="600">
</p>


Examples of the clusters found by DeepDPM on the ImageNet Dataset:
![Examples of the clusters found by DeepDPM on the ImageNet dataset](images/imagenet_cluster_examples.jpg?raw=true "Examples of the clusters found by DeepDPM on the ImageNet dataset")

## Introduction
DeepDPM is a nonparametric deep-clustering method which unlike most deep clustering methods, does not require knowing the number of clusters, K; rather, it infers it as a part of the overall learning. Using a split/merge framework to change the clusters number adaptively and a novel loss, our proposed method outperforms existing (both classical and deep) nonparametric methods.

While the few existing deep nonparametric methods lack scalability, we show ours by being the first such method that reports its performance on ImageNet.

**Key phases:**
- **Warmup**: Network trains with frozen GMM to stabilize representations
- **Split**: SubclusterNet proposes splits, accepted via Bayesian Hastings ratio
- **Merge**: Nearest cluster pairs proposed, accepted if marginal likelihood improves
- **Alternation**: Splits and merges alternate to prevent oscillation

## Installation

### Requirements

- **Python 3.10+**

```bash
# Core dependencies
torch>=1.10.0
numpy>=1.20.0
scikit-learn>=0.24.0
scipy>=1.7.0
pyyaml>=5.4.0
kmeans-pytorch>=0.3    # GPU-accelerated K-means
tqdm>=4.60.0           # Progress bars (required by kmeans-pytorch)
```

### Setup

```bash
# Create conda environment with Python 3.10+
conda create -n deepdpm python=3.10 -y
conda activate deepdpm

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training on Synthetic Data

```bash
python scripts/train_synthetic.py
```

Expected output:
```
Creating synthetic GMM data...
Data shape: torch.Size([2000, 10])
True number of clusters: 5

======================================================================
DeepDPM Training
======================================================================
Device: cuda
Initial K: 1
Num epochs: 500
======================================================================
Gathering data and initializing with K-means...
Data shape: torch.Size([2000, 10])
Initializing with K=1
Initialization complete. K=1
Epoch   1/500: K= 1, cluster_loss=0.0000, sub_loss=0.0000, splits=0, merges=0, time=0.3s
...
Epoch  30/500:
  K: 2
  Cluster loss: 0.0000
  Subcluster loss: 115.4761
  Splits: 1
  Accuracy: 0.2000
  NMI: 0.0000
  ARI: 0.0000
  Time: 0.2s
...
Epoch 150/500:
  K: 4
  Cluster loss: 0.0000
  Subcluster loss: 64.6341
  Splits: 1
  Accuracy: 0.8000
  NMI: 0.9098
  ARI: 0.7823
  Time: 0.2s
...
Epoch 500/500:
  K: 5
  Cluster loss: 0.0000
  Subcluster loss: 10.8619
  Accuracy: 1.0000
  NMI: 1.0000
  ARI: 1.0000
  Time: 0.3s
======================================================================
Training completed!
Final K: 5
======================================================================

======================================================================
FINAL RESULTS
======================================================================
True K: 5
Inferred K: 5
Final Accuracy: 1.0000
Final NMI: 1.0000
Final ARI: 1.0000
======================================================================
```

### Training with Custom Config

```python
import torch
from data import create_synthetic_gmm_data, create_dataloader
from training import DeepDPMTrainer
from configs import TrainingConfig
from utils import set_seed

# Load config from YAML
config = TrainingConfig.from_yaml("configs/experiments/synthetic.yaml")

# Or create programmatically
config = TrainingConfig(
    input_dim=10,
    hidden_dims=[50],
    init_k=1,
    num_epochs=500,  # Paper uses 500 epochs
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Set seed
set_seed(config.seed)

# Create data
data, labels = create_synthetic_gmm_data(n_samples=2000, n_clusters=5)
train_loader = create_dataloader(data, batch_size=config.batch_size)

# Train
trainer = DeepDPMTrainer(config)
trainer.train(train_loader, true_labels=labels)
```

### Available Scripts

**Python Scripts:**
| Script | Description |
|--------|-------------|
| `train_from_config.py` | Universal training script - trains DeepDPM with config file and embeddings |
| `train_synthetic.py` | Synthetic GMM training with generated data |

**Shell Scripts (Experiments):**
| Script | Description |
|--------|-------------|
| `run_synthetic.sh` | Run synthetic GMM experiment |
| `run_mnist.sh` | Run MNIST experiment |
| `run_mnist_imbalanced.sh` | Run MNIST with imbalanced clusters |
| `run_fashion_mnist.sh` | Run Fashion-MNIST experiment |
| `run_fashion_mnist_imbalanced.sh` | Run Fashion-MNIST with imbalanced clusters |
| `run_usps.sh` | Run USPS experiment |
| `run_usps_imbalanced.sh` | Run USPS with imbalanced clusters |
| `run_stl10.sh` | Run STL-10 experiment |
| `run_imagenet50.sh` | Run ImageNet-50 experiment |
| `run_imagenet50_imbalanced.sh` | Run ImageNet-50 with imbalanced clusters |


**Key training features:**
- **Split/merge alternation**: Prevents consecutive splits or merges for stability
- **GMM warmup**: GMM parameters frozen for initial epochs to let network stabilize
- **Freeze periods**: After split/merge, GMM frozen for N epochs to allow adaptation
- **LR scheduler skip**: Learning rate scheduler skipped during freeze periods

## Configuration

### YAML Configuration

Example (`configs/experiments/mnist.yaml`):

```yaml
# Configuration for MNIST (after UMAP embedding to 10D)

model:
  input_dim: 10  # After UMAP reduction
  hidden_dims: [50]
  init_k: 1

training:
  batch_size: 128
  num_epochs: 500
  cluster_lr: 0.0005
  subcluster_lr: 0.005

  cluster_loss_type: "KL_GMM_2"
  subcluster_loss_type: "isotropic"

  start_sub_clustering: 45
  start_splitting: 55
  start_merging: 55
  split_merge_every_n_epochs: 30
  eval_every_n_epochs: 10

prior:
  kappa: 0.0001
  nu_offset: 2
  psi_scale: 0.005

split:
  alpha: 10.0
  stochastic_accept: true
  min_cluster_size: 2

merge:
  k_nearest: 3
  proposal_method: "kmeans"
  stochastic_accept: true

seed: 42
device: "cuda"
log_dir: "./logs/mnist"
checkpoint_dir: "./checkpoints/mnist"
save_every_n_epochs: 50
verbose: true
```

### Key Hyperparameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `init_k` | Initial number of clusters | 1 |
| `cluster_lr` | Learning rate for cluster net | 0.0005 |
| `subcluster_lr` | Learning rate for subcluster net | 0.005 |
| `start_sub_clustering` | Epoch to start subclustering | 45 |
| `start_splitting` | Epoch to start splits | 55 |
| `alpha` | DP concentration (higher = more clusters) | 10.0 |
| `kappa` | Prior pseudocount (lower = weaker prior) | 0.0001 |

## Evaluation Metrics

- **Accuracy (ACC)**: Clustering accuracy with optimal Hungarian assignment
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index

All metrics handle different K between prediction and ground truth.

## Expected Results

| Dataset | GT K | Inferred K | ACC | NMI | ARI |
|---------|------|------------|-----|-----|-----|
| Synthetic (5 clusters) | 5 | 5 | 1.00 | 1.00 | 1.00 |
| MNIST | 10 | 10±0 | 0.98±0.00 | 0.94±0.00 | 0.95±0.00 |
| Fashion-MNIST | 10 | 10.2±0.79 | 0.62±0.03 | 0.68±0.01 | 0.51±0.02 |

## Citation

```bibtex
@inproceedings{ronen2022deepdpm,
  title={DeepDPM: Deep Clustering With an Unknown Number of Clusters},
  author={Ronen, Meitar and Finder, Shahaf E and Freifeld, Oren},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9861--9870},
  year={2022}
}
```

## License

See [LICENSE](LICENSE).

## Contributing

This is a clean reimplementation for educational and research purposes. Contributions are welcome:
- Bug fixes
- Performance improvements
- Additional datasets
- Documentation improvements

## Contact

For questions about this implementation, please open an issue on GitHub.

