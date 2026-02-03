"""Subcluster assignment loss functions."""

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reimplementation.gmm.parameters import GMMParameters


def subcluster_isotropic_loss(
    subcluster_resp: torch.Tensor,
    gmm_params: "GMMParameters",
    data: torch.Tensor,
    cluster_assignments: torch.Tensor  # Kept for API compatibility, not used
) -> torch.Tensor:
    """Isotropic loss for subclustering.

    Vectorized implementation.

    Formula: mean(sum_j r_ij * ||x_i - mu_j||^2)
    where j ∈ [0, 2*K) are all subclusters, but r_ij = 0 for irrelevant subclusters

    Args:
        subcluster_resp: (N, 2*K) subcluster responsibilities
                         This should come from compute_subcluster_resp() which applies
                         masked softmax with temperature scaling.
        gmm_params: Current GMM parameters
        data: (N, d) input features
        cluster_assignments: (N,) hard cluster assignments - kept for API compatibility

    Returns:
        Scalar loss: weighted squared distances to subcluster means
    """
    N, d = data.shape
    K = gmm_params.k

    # subcluster_resp is already softmaxed (done by compute_subcluster_resp in trainer)
    subcluster_probs = subcluster_resp

    # Vectorized implementation
    # Repeat data: (N, d) -> (N, 2*K, d) -> (N*2*K, d)
    data_repeated = data.unsqueeze(1).repeat(1, 2*K, 1).view(-1, d)  # (N*2*K, d)

    # Repeat mus_sub: (2*K, d) -> (N, 2*K, d) -> (N*2*K, d)
    mus_repeated = gmm_params.mus_sub.unsqueeze(0).repeat(N, 1, 1).view(-1, d)  # (N*2*K, d)

    # Flatten responsibilities: (N, 2*K) -> (N*2*K,)
    responsibilities_flat = subcluster_probs.flatten()  # (N*2*K,)

    # Compute squared distances
    squared_dists = torch.sum((data_repeated - mus_repeated) ** 2, dim=1)  # (N*2*K,)

    # Weighted distances
    weighted_dists = responsibilities_flat * squared_dists  # (N*2*K,)

    # Average over all points
    return weighted_dists.sum() / float(N)
