"""Dataset utilities for DeepDPM."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Optional
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


def transform_embeddings(
    data: torch.Tensor,
    transform_type: str = "normalize"
) -> torch.Tensor:
    """Transform embeddings for DeepDPM preprocessing.

    Args:
        data: (N, d) input data tensor
        transform_type: One of:
            - "normalize": L2 normalize each sample (DEFAULT)
            - "min_max": Min-max scaling to [0, 1]
            - "standard": Z-score standardization
            - "None" or None: No transformation

    Returns:
        Transformed data tensor
    """
    if transform_type in (None, "None"):
        return data

    # Convert to numpy for sklearn
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data

    if transform_type == "normalize":
        # L2 normalization per sample (each row has unit norm)
        # This is the DEFAULT
        transformed = Normalizer().fit_transform(data_np)
    elif transform_type == "min_max":
        transformed = MinMaxScaler().fit_transform(data_np)
    elif transform_type == "standard":
        transformed = StandardScaler().fit_transform(data_np)
    elif transform_type == "standard_normalize":
        # Standard scale then normalize
        transformed = StandardScaler().fit_transform(data_np)
        transformed = Normalizer().fit_transform(transformed)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    return torch.from_numpy(transformed).float()


def create_synthetic_gmm_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_clusters: int = 5,
    cluster_std: float = 1.0,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data from a Gaussian mixture model.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_clusters: True number of clusters
        cluster_std: Standard deviation of clusters
        random_state: Random seed

    Returns:
        Tuple of (data, labels)
            data: (n_samples, n_features) tensor
            labels: (n_samples,) tensor of cluster labels
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )

    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def load_embeddings_from_file(
    file_path: str,
    labels_path: Optional[str] = None,
    transform_type: str = "normalize"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load pre-computed embeddings from file.

    Args:
        file_path: Path to embeddings file (.npy or .pt)
        labels_path: Optional path to labels file
        transform_type: Data transformation type (default: "normalize")

    Returns:
        Tuple of (embeddings, labels)
            embeddings: (N, d) tensor - transformed if transform_type specified
            labels: (N,) tensor or None
    """
    # Load embeddings
    if file_path.endswith('.npy'):
        embeddings = torch.from_numpy(np.load(file_path)).float()
    elif file_path.endswith('.pt') or file_path.endswith('.pth'):
        embeddings = torch.load(file_path, weights_only=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Ensure embeddings are float tensors
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.from_numpy(embeddings).float()
    elif embeddings.dtype != torch.float32:
        embeddings = embeddings.float()

    # Apply transformation (CRITICAL: default is "normalize")
    if transform_type:
        embeddings = transform_embeddings(embeddings, transform_type)

    # Load labels if provided
    labels = None
    if labels_path is not None:
        if labels_path.endswith('.npy'):
            labels = torch.from_numpy(np.load(labels_path)).long()
        elif labels_path.endswith('.pt') or labels_path.endswith('.pth'):
            labels = torch.load(labels_path, weights_only=False)
        else:
            raise ValueError(f"Unsupported file format: {labels_path}")

    return embeddings, labels


def create_dataloader(
    data: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader from tensors.

    Args:
        data: (N, d) data tensor
        labels: Optional (N,) labels tensor
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    if labels is not None:
        dataset = TensorDataset(data, labels)
    else:
        dataset = TensorDataset(data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
