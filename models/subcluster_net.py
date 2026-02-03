"""SubclusterNet: Network for 2-way subclustering within each cluster."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubclusterNet(nn.Module):
    """Subclustering network with K independent 2-way classifiers.

    Architecture: [input_dim -> 50*K -> 2*K]
    For each cluster k, outputs softmax probabilities over 2 subclusters.
    Output indexing: cluster k has subclusters at indices [2k, 2k+1]

    Key property: Independent gradient flow for each cluster's subclusters.
    Only subclusters for points assigned to cluster k receive gradients.
    """

    def __init__(self, input_dim: int, init_k: int, softmax_norm: float = 1.0):
        """Initialize SubclusterNet.

        Args:
            input_dim: Dimension of input features
            init_k: Initial number of clusters
            softmax_norm: Temperature scaling for softmax (higher = sharper).
                         Default is 1.0.
        """
        super().__init__()

        self.input_dim = input_dim
        self._k = init_k
        self.softmax_norm = softmax_norm  # Temperature scaling for softmax

        # Two-layer architecture
        # First layer: input_dim -> 50*K (scales with K)
        self.fc1 = nn.Linear(input_dim, 50 * init_k)

        # Second layer: 50*K -> 2*K (one 2-way classifier per cluster)
        self.fc2 = nn.Linear(50 * init_k, 2 * init_k)

        # CRITICAL: Apply gradient masking to ensure cluster independence
        self._apply_gradient_mask()

    @property
    def k(self) -> int:
        """Current number of clusters."""
        return self._k

    def _apply_gradient_mask(self):
        """Apply gradient masking to fc2 weights to ensure cluster independence.

        Uses gradient hooks to zero out weights and gradients between different
        subclustering networks.

        For each cluster k:
        - Allow connections from hidden units [50*k : 50*(k+1)] to output units [2*k : 2*(k+1)]
        - Mask all other connections to zero

        This ensures that:
        1. Each cluster's subclustering network is completely independent
        2. After splits, new clusters learn independently without interference
        3. No gradient flow between different clusters' subclustering parameters
        """
        # Create gradient mask: zeros except for diagonal blocks
        # Shape: (50*K, 2*K) - same as fc2.weight transposed
        gradient_mask = torch.zeros(50 * self._k, 2 * self._k)

        for k in range(self._k):
            # Allow connections from hidden units [50*k : 50*(k+1)]
            # to output units [2*k : 2*(k+1)]
            gradient_mask[50*k : 50*(k+1), 2*k : 2*(k+1)] = 1

        # Store mask for potential inspection
        self.gradient_mask = gradient_mask

        with torch.no_grad():
            # Zero out masked weights initially (move mask to same device as weight)
            self.fc2.weight.data *= gradient_mask.T.to(self.fc2.weight.device)

        # Register gradient hook to mask gradients during backprop
        # Note: weight is (2*K, 50*K), so we need to transpose the mask
        def gradient_mask_hook(grad):
            return grad * gradient_mask.T.to(device=grad.device)

        # Remove any existing hooks first
        if hasattr(self.fc2.weight, '_gradient_hook_handle'):
            self.fc2.weight._gradient_hook_handle.remove()

        # Register new hook
        hook_handle = self.fc2.weight.register_hook(gradient_mask_hook)
        # Store handle so we can remove it later if needed
        self.fc2.weight._gradient_hook_handle = hook_handle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: (N, input_dim) input features

        Returns:
            (N, 2*K) raw logits for all subclusters (no softmax applied)
        """
        # Forward through network
        h = F.relu(self.fc1(x))  # (N, 50*K)
        logits = self.fc2(h)  # (N, 2*K)

        return logits

    def forward_all_subclusters(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all 2K subcluster probabilities (for split proposals).

        Args:
            x: (N, input_dim) input features

        Returns:
            (N, 2*K) soft assignments for all subclusters

        Note:
            This is used during split proposal evaluation to assess
            how well each cluster can be divided into subclusters.
        """
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)  # (N, 2*K)

        # Apply softmax for each cluster's 2 subclusters independently
        # Reshape to (N, K, 2), apply softmax with temperature over last dim, reshape back
        batch_size = x.shape[0]
        logits_reshaped = logits.view(batch_size, self._k, 2)
        probs_reshaped = F.softmax(logits_reshaped * self.softmax_norm, dim=2)
        return probs_reshaped.view(batch_size, 2 * self._k)

    def split_cluster(self, cluster_idx: int, init_weights: str = "random"):
        """Add subclustering nets for two new clusters after split.

        When cluster k splits into k_1 and k_2:
        - Old: K clusters with 2K subclusters
        - New: K+1 clusters with 2(K+1) subclusters

        Args:
            cluster_idx: Index of cluster that was split
            init_weights: How to initialize new subcluster output weights
                - "same": Duplicate parent cluster's subcluster weights
                - "random": Random uniform[-1, 1]
        """
        if cluster_idx < 0 or cluster_idx >= self._k:
            raise ValueError(f"Invalid cluster_idx {cluster_idx}, K={self._k}")

        with torch.no_grad():
            # Get current weights
            old_fc1_weight = self.fc1.weight.data  # (50*K, input_dim)
            old_fc1_bias = self.fc1.bias.data  # (50*K,)
            old_fc2_weight = self.fc2.weight.data  # (2*K, 50*K)
            old_fc2_bias = self.fc2.bias.data  # (2*K,)

            # Create new layers with K+1 clusters
            new_k = self._k + 1
            new_fc1 = nn.Linear(self.input_dim, 50 * new_k)
            new_fc2 = nn.Linear(50 * new_k, 2 * new_k)

            # Move to same device
            new_fc1 = new_fc1.to(old_fc1_weight.device)
            new_fc2 = new_fc2.to(old_fc2_weight.device)

            # === Update fc1: (50*K, input_dim) -> (50*(K+1), input_dim) ===
            # Add 50 new units for the new cluster
            new_fc1_weight = new_fc1.weight.data
            new_fc1_bias = new_fc1.bias.data

            # Copy weights before split cluster
            split_start = cluster_idx * 50
            split_end = (cluster_idx + 1) * 50

            new_fc1_weight[:split_start] = old_fc1_weight[:split_start]
            new_fc1_bias[:split_start] = old_fc1_bias[:split_start]

            # Initialize weights for split cluster based on init_weights mode
            if init_weights == "same":
                # Duplicate weights for split cluster (2 copies)
                new_fc1_weight[split_start:split_start+50] = old_fc1_weight[split_start:split_end]
                new_fc1_bias[split_start:split_start+50] = old_fc1_bias[split_start:split_end]
                new_fc1_weight[split_start+50:split_start+100] = old_fc1_weight[split_start:split_end]
                new_fc1_bias[split_start+50:split_start+100] = old_fc1_bias[split_start:split_end]
            else:  # init_weights == "random"
                device = old_fc1_weight.device
                new_fc1_weight[split_start:split_start+100] = \
                    torch.FloatTensor(100, self.input_dim).uniform_(-1., 1.).to(device)
                new_fc1_bias[split_start:split_start+100] = 0.0

            # Copy weights after split cluster
            new_fc1_weight[split_start+100:] = old_fc1_weight[split_end:]
            new_fc1_bias[split_start+100:] = old_fc1_bias[split_end:]

            # === Update fc2: (2*K, 50*K) -> (2*(K+1), 50*(K+1)) ===
            # Add 2 output units and expand input dimension
            new_fc2_weight = new_fc2.weight.data
            new_fc2_bias = new_fc2.bias.data

            # Output dimension: add 2 subclusters for new cluster
            subcluster_start = cluster_idx * 2
            subcluster_end = (cluster_idx + 1) * 2

            # Input dimension: expand to account for new fc1 units
            # We need to map old (2*K, 50*K) to new (2*(K+1), 50*(K+1))

            # Copy output rows before split
            new_fc2_weight[:subcluster_start, :split_start] = \
                old_fc2_weight[:subcluster_start, :split_start]
            new_fc2_bias[:subcluster_start] = old_fc2_bias[:subcluster_start]

            # Initialize new subcluster outputs
            if init_weights == "same":
                # Duplicate parent's subcluster weights for new clusters
                for i in range(2):
                    new_fc2_weight[subcluster_start + i, :split_start] = \
                        old_fc2_weight[subcluster_start + i, :split_start]
                    new_fc2_weight[subcluster_start + i, split_start:split_start+50] = \
                        old_fc2_weight[subcluster_start + i, split_start:split_end]
                    new_fc2_weight[subcluster_start + i, split_start+50:split_start+100] = \
                        old_fc2_weight[subcluster_start + i, split_start:split_end]
                    new_fc2_weight[subcluster_start + i, split_start+100:] = \
                        old_fc2_weight[subcluster_start + i, split_end:]
                    new_fc2_bias[subcluster_start + i] = old_fc2_bias[subcluster_start + i]

                for i in range(2):
                    new_fc2_weight[subcluster_start + 2 + i, :split_start] = \
                        old_fc2_weight[subcluster_start + i, :split_start]
                    new_fc2_weight[subcluster_start + 2 + i, split_start:split_start+50] = \
                        old_fc2_weight[subcluster_start + i, split_start:split_end]
                    new_fc2_weight[subcluster_start + 2 + i, split_start+50:split_start+100] = \
                        old_fc2_weight[subcluster_start + i, split_start:split_end]
                    new_fc2_weight[subcluster_start + 2 + i, split_start+100:] = \
                        old_fc2_weight[subcluster_start + i, split_end:]
                    new_fc2_bias[subcluster_start + 2 + i] = old_fc2_bias[subcluster_start + i]

            elif init_weights == "random":
                # Random weights, zero biases
                device = old_fc2_weight.device
                for i in range(4):
                    new_fc2_weight[subcluster_start + i] = \
                        torch.FloatTensor(new_fc2_weight[subcluster_start + i].shape).uniform_(-1., 1.).to(device)
                    new_fc2_bias[subcluster_start + i] = 0.0
            else:
                raise ValueError(f"Unknown init_weights mode: {init_weights}. Must be 'same' or 'random'.")

            # Copy output rows after split
            old_after_start = subcluster_end
            new_after_start = subcluster_start + 4

            new_fc2_weight[new_after_start:, :split_start] = \
                old_fc2_weight[old_after_start:, :split_start]
            new_fc2_weight[new_after_start:, split_start:split_start+50] = \
                old_fc2_weight[old_after_start:, split_start:split_end]
            new_fc2_weight[new_after_start:, split_start+50:split_start+100] = \
                old_fc2_weight[old_after_start:, split_start:split_end]
            new_fc2_weight[new_after_start:, split_start+100:] = \
                old_fc2_weight[old_after_start:, split_end:]
            new_fc2_bias[new_after_start:] = old_fc2_bias[old_after_start:]

            # Replace layers
            self.fc1 = new_fc1
            self.fc2 = new_fc2
            self._k = new_k

        # CRITICAL: Reapply gradient masking for new architecture
        self._apply_gradient_mask()

    def merge_clusters(self, cluster_idx1: int, cluster_idx2: int):
        """Remove and reinitialize subclustering nets after merge.

        When clusters k1 and k2 merge:
        - Old: K clusters with 2K subclusters
        - New: K-1 clusters with 2(K-1) subclusters

        Args:
            cluster_idx1: First cluster index
            cluster_idx2: Second cluster index
        """
        if cluster_idx1 < 0 or cluster_idx1 >= self._k:
            raise ValueError(f"Invalid cluster_idx1 {cluster_idx1}, K={self._k}")
        if cluster_idx2 < 0 or cluster_idx2 >= self._k:
            raise ValueError(f"Invalid cluster_idx2 {cluster_idx2}, K={self._k}")
        if cluster_idx1 == cluster_idx2:
            raise ValueError("Cannot merge cluster with itself")

        # Ensure cluster_idx1 < cluster_idx2
        if cluster_idx1 > cluster_idx2:
            cluster_idx1, cluster_idx2 = cluster_idx2, cluster_idx1

        with torch.no_grad():
            old_fc1_weight = self.fc1.weight.data
            old_fc1_bias = self.fc1.bias.data
            old_fc2_weight = self.fc2.weight.data
            old_fc2_bias = self.fc2.bias.data

            new_k = self._k - 1
            new_fc1 = nn.Linear(self.input_dim, 50 * new_k)
            new_fc2 = nn.Linear(50 * new_k, 2 * new_k)

            new_fc1 = new_fc1.to(old_fc1_weight.device)
            new_fc2 = new_fc2.to(old_fc2_weight.device)

            # === Update fc1: Remove 50 units for one cluster ===
            new_fc1_weight = new_fc1.weight.data
            new_fc1_bias = new_fc1.bias.data

            # Merged cluster at cluster_idx1, remove cluster_idx2
            # Average the weights for the merged cluster
            merge1_start = cluster_idx1 * 50
            merge1_end = (cluster_idx1 + 1) * 50
            merge2_start = cluster_idx2 * 50
            merge2_end = (cluster_idx2 + 1) * 50

            # Before first cluster
            new_fc1_weight[:merge1_start] = old_fc1_weight[:merge1_start]
            new_fc1_bias[:merge1_start] = old_fc1_bias[:merge1_start]

            # Merged cluster (average)
            new_fc1_weight[merge1_start:merge1_end] = \
                (old_fc1_weight[merge1_start:merge1_end] +
                 old_fc1_weight[merge2_start:merge2_end]) / 2
            new_fc1_bias[merge1_start:merge1_end] = \
                (old_fc1_bias[merge1_start:merge1_end] +
                 old_fc1_bias[merge2_start:merge2_end]) / 2

            # Between clusters
            new_fc1_weight[merge1_end:merge2_start] = \
                old_fc1_weight[merge1_end:merge2_start]
            new_fc1_bias[merge1_end:merge2_start] = \
                old_fc1_bias[merge1_end:merge2_start]

            # After second cluster
            new_fc1_weight[merge2_start:] = old_fc1_weight[merge2_end:]
            new_fc1_bias[merge2_start:] = old_fc1_bias[merge2_end:]

            # === Update fc2: Similar process for output ===
            new_fc2_weight = new_fc2.weight.data
            new_fc2_bias = new_fc2.bias.data

            sub1_start = cluster_idx1 * 2
            sub1_end = (cluster_idx1 + 1) * 2
            sub2_start = cluster_idx2 * 2
            sub2_end = (cluster_idx2 + 1) * 2

            # Map input dimensions (account for removed fc1 units)

            # Before first subcluster pair
            new_fc2_weight[:sub1_start, :merge1_start] = \
                old_fc2_weight[:sub1_start, :merge1_start]
            new_fc2_bias[:sub1_start] = old_fc2_bias[:sub1_start]

            # Merged subclusters (average)
            for i in range(2):
                new_fc2_weight[sub1_start + i, :merge1_start] = \
                    (old_fc2_weight[sub1_start + i, :merge1_start] +
                     old_fc2_weight[sub2_start + i, :merge1_start]) / 2
                new_fc2_weight[sub1_start + i, merge1_start:merge1_end] = \
                    (old_fc2_weight[sub1_start + i, merge1_start:merge1_end] +
                     old_fc2_weight[sub2_start + i, merge2_start:merge2_end]) / 2
                new_fc2_weight[sub1_start + i, merge1_end:merge2_start] = \
                    (old_fc2_weight[sub1_start + i, merge1_end:merge2_start] +
                     old_fc2_weight[sub2_start + i, merge1_end:merge2_start]) / 2
                new_fc2_weight[sub1_start + i, merge2_start:] = \
                    (old_fc2_weight[sub1_start + i, merge2_end:] +
                     old_fc2_weight[sub2_start + i, merge2_end:]) / 2
                new_fc2_bias[sub1_start + i] = \
                    (old_fc2_bias[sub1_start + i] + old_fc2_bias[sub2_start + i]) / 2

            # Between subclusters (map input dims carefully)
            for row in range(sub1_end, sub2_start):
                new_fc2_weight[row, :merge1_start] = old_fc2_weight[row, :merge1_start]
                new_fc2_weight[row, merge1_start:merge1_end] = \
                    old_fc2_weight[row, merge1_start:merge1_end]
                new_fc2_weight[row, merge1_end:merge2_start] = \
                    old_fc2_weight[row, merge1_end:merge2_start]
                new_fc2_weight[row, merge2_start:] = old_fc2_weight[row, merge2_end:]
                new_fc2_bias[row] = old_fc2_bias[row]

            # After second subcluster pair (map input dims)
            for i, old_row in enumerate(range(sub2_end, 2 * self._k)):
                new_row = sub2_start + i
                new_fc2_weight[new_row, :merge1_start] = old_fc2_weight[old_row, :merge1_start]
                new_fc2_weight[new_row, merge1_start:merge1_end] = \
                    old_fc2_weight[old_row, merge1_start:merge1_end]
                new_fc2_weight[new_row, merge1_end:merge2_start] = \
                    old_fc2_weight[old_row, merge1_end:merge2_start]
                new_fc2_weight[new_row, merge2_start:] = old_fc2_weight[old_row, merge2_end:]
                new_fc2_bias[new_row] = old_fc2_bias[old_row]

            # Apply symmetry-breaking bias to merged cluster's subclusters
            new_fc2_bias[2*cluster_idx1] = 0.5       # Slightly favor subcluster 0
            new_fc2_bias[2*cluster_idx1 + 1] = -0.5  # Slightly disfavor subcluster 1

            self.fc1 = new_fc1
            self.fc2 = new_fc2
            self._k = new_k

        # CRITICAL: Reapply gradient masking for new architecture
        self._apply_gradient_mask()
