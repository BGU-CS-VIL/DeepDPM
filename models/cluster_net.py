"""ClusterNet: MLP classifier with dynamic number of output clusters."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ClusterNet(nn.Module):
    """MLP classifier with dynamic output dimension for K clusters.

    Architecture: [input_dim -> hidden[0] -> ... -> hidden[-1] -> K]
    Output: Softmax probabilities over K clusters

    The number of clusters K can change dynamically during training via
    split_cluster() and merge_clusters() operations.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], init_k: int,
                 softmax_norm: float = 1.0):
        """Initialize ClusterNet.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions (e.g., [50])
            init_k: Initial number of clusters
            softmax_norm: Temperature scaling for softmax (default: 1.0)
                         Higher values make probabilities sharper, lower values softer.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self._k = init_k
        self.softmax_norm = softmax_norm

        # Build hidden layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()
        self.hidden_out_dim = prev_dim

        # Output layer (dynamically updated)
        self.output_layer = nn.Linear(self.hidden_out_dim, init_k)

    @property
    def k(self) -> int:
        """Current number of clusters."""
        return self._k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: (N, input_dim) input features

        Returns:
            (N, K) soft cluster assignments (softmax probabilities)
        """
        h = self.hidden_layers(x)
        logits = self.output_layer(h)
        logits = logits * self.softmax_norm
        return F.softmax(logits, dim=1)

    def split_cluster(
        self,
        cluster_idx: int,
        init_weights: str = "same",
    ):
        """Split cluster_idx into two new clusters.

        This operation:
        1. Increases K by 1
        2. Duplicates the output layer weights for cluster_idx
        3. Initializes new weights based on init_weights strategy

        Args:
            cluster_idx: Index of cluster to split (0 <= cluster_idx < K)
            init_weights: Initialization strategy:
                - "same": Duplicate weights exactly
                - "random": Reinitialize randomly with small noise

        Updates:
            - self._k increases by 1
            - self.output_layer expanded to (hidden_out_dim, K+1)
            - Cluster ordering: [..., k-1, k_new1, k_new2, k+1, ...]
        """
        if cluster_idx < 0 or cluster_idx >= self._k:
            raise ValueError(f"Invalid cluster_idx {cluster_idx}, K={self._k}")

        # Get current weights and bias
        with torch.no_grad():
            old_weight = self.output_layer.weight.data  # (K, hidden_out_dim)
            old_bias = self.output_layer.bias.data  # (K,)

            # Create new output layer with K+1 units
            new_output_layer = nn.Linear(self.hidden_out_dim, self._k + 1)
            new_output_layer = new_output_layer.to(old_weight.device)

            # Copy weights for all clusters except cluster_idx
            # New layout: [0, ..., cluster_idx-1, cluster_idx_1, cluster_idx_2, cluster_idx+1, ..., K-1]
            new_weight = new_output_layer.weight.data
            new_bias = new_output_layer.bias.data

            # Copy clusters before split point
            new_weight[:cluster_idx] = old_weight[:cluster_idx]
            new_bias[:cluster_idx] = old_bias[:cluster_idx]

            # Initialize two new clusters
            if init_weights == "same":
                # Copy the same weights from parent cluster to both children
                new_weight[cluster_idx] = old_weight[cluster_idx].clone()
                new_weight[cluster_idx + 1] = old_weight[cluster_idx].clone()
                new_bias[cluster_idx] = old_bias[cluster_idx].clone()
                new_bias[cluster_idx + 1] = old_bias[cluster_idx].clone()

            elif init_weights == "random":
                # Completely reinitialize with uniform random weights in [-1, 1]
                device = old_weight.device
                new_weight[cluster_idx] = torch.FloatTensor(old_weight[cluster_idx].shape).uniform_(-1., 1.).to(device)
                new_weight[cluster_idx + 1] = torch.FloatTensor(old_weight[cluster_idx].shape).uniform_(-1., 1.).to(device)
                new_bias[cluster_idx] = torch.FloatTensor(old_bias[cluster_idx].shape).uniform_(-1., 1.).to(device)
                new_bias[cluster_idx + 1] = torch.FloatTensor(old_bias[cluster_idx].shape).uniform_(-1., 1.).to(device)

            else:
                raise ValueError(f"Unknown init_weights: {init_weights}")

            # Copy clusters after split point
            new_weight[cluster_idx + 2:] = old_weight[cluster_idx + 1:]
            new_bias[cluster_idx + 2:] = old_bias[cluster_idx + 1:]

            # Replace output layer
            self.output_layer = new_output_layer
            self._k += 1

    def merge_clusters(self, cluster_idx1: int, cluster_idx2: int):
        """Merge two clusters into one.

        This operation:
        1. Decreases K by 1
        2. Removes one cluster's output unit
        3. Initializes merged cluster with weighted average

        Args:
            cluster_idx1: First cluster index
            cluster_idx2: Second cluster index

        Note:
            The merged cluster will be at position min(cluster_idx1, cluster_idx2)
        """
        if cluster_idx1 < 0 or cluster_idx1 >= self._k:
            raise ValueError(f"Invalid cluster_idx1 {cluster_idx1}, K={self._k}")
        if cluster_idx2 < 0 or cluster_idx2 >= self._k:
            raise ValueError(f"Invalid cluster_idx2 {cluster_idx2}, K={self._k}")
        if cluster_idx1 == cluster_idx2:
            raise ValueError("Cannot merge cluster with itself")

        # Ensure cluster_idx1 < cluster_idx2 for consistent ordering
        if cluster_idx1 > cluster_idx2:
            cluster_idx1, cluster_idx2 = cluster_idx2, cluster_idx1

        with torch.no_grad():
            old_weight = self.output_layer.weight.data  # (K, hidden_out_dim)
            old_bias = self.output_layer.bias.data  # (K,)

            # Create new output layer with K-1 units
            new_output_layer = nn.Linear(self.hidden_out_dim, self._k - 1)
            new_output_layer = new_output_layer.to(old_weight.device)

            new_weight = new_output_layer.weight.data
            new_bias = new_output_layer.bias.data

            # Merged cluster at cluster_idx1 position
            # Simple average (could be weighted by cluster sizes in future)
            merged_weight = (old_weight[cluster_idx1] + old_weight[cluster_idx2]) / 2
            merged_bias = (old_bias[cluster_idx1] + old_bias[cluster_idx2]) / 2

            # Build new weight matrix
            # Layout: [0, ..., cluster_idx1-1, merged, cluster_idx1+1, ...,
            #          cluster_idx2-1, cluster_idx2+1, ..., K-1]

            # Before first cluster
            new_weight[:cluster_idx1] = old_weight[:cluster_idx1]
            new_bias[:cluster_idx1] = old_bias[:cluster_idx1]

            # Merged cluster
            new_weight[cluster_idx1] = merged_weight
            new_bias[cluster_idx1] = merged_bias

            # Between clusters
            new_weight[cluster_idx1 + 1:cluster_idx2] = \
                old_weight[cluster_idx1 + 1:cluster_idx2]
            new_bias[cluster_idx1 + 1:cluster_idx2] = \
                old_bias[cluster_idx1 + 1:cluster_idx2]

            # After second cluster
            new_weight[cluster_idx2:] = old_weight[cluster_idx2 + 1:]
            new_bias[cluster_idx2:] = old_bias[cluster_idx2 + 1:]

            # Replace output layer
            self.output_layer = new_output_layer
            self._k -= 1
