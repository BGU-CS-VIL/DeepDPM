"""Cluster assignment loss functions."""

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reimplementation.gmm.parameters import GMMParameters


def kl_gmm_loss(
    cluster_probs: torch.Tensor,
    gmm_params: "GMMParameters",
    data: torch.Tensor
) -> torch.Tensor:
    """KL divergence between GMM responsibilities and network predictions.

    This is the main loss function from the paper (Eq. 6).
    Computes KL(GMM || Network) which encourages the network to
    COVER ALL modes of the GMM (mode-covering behavior).

    Args:
        cluster_probs: (N, K) soft assignments from ClusterNet
        gmm_params: Current GMM parameters
        data: (N, d) input features

    Returns:
        Scalar loss: batchmean KL(gmm_responsibilities || network)
    """
    # Compute GMM E-step responsibilities (target)
    gmm_log_resp = gmm_params.compute_log_responsibilities(data)  # (N, K)

    # Normalize using log-sum-exp
    max_values, _ = gmm_log_resp.max(dim=1, keepdim=True)
    gmm_log_resp_norm = gmm_log_resp - torch.log(
        torch.exp(gmm_log_resp - max_values).sum(dim=1, keepdim=True)
    ) - max_values
    gmm_resp = torch.exp(gmm_log_resp_norm)

    # Add epsilon and renormalize
    eps = 0.00001
    gmm_resp = (gmm_resp + eps) / (gmm_resp + eps).sum(dim=1, keepdim=True)
    cluster_probs_norm = (cluster_probs + eps) / (cluster_probs + eps).sum(dim=1, keepdim=True)

    return F.kl_div(
        torch.log(cluster_probs_norm),  # input: log(network_probs)
        gmm_resp,                       # target: gmm_probs
        reduction='batchmean'
    )


def isotropic_loss(
    cluster_probs: torch.Tensor,
    gmm_params: "GMMParameters",
    data: torch.Tensor
) -> torch.Tensor:
    """Isotropic distance loss (simpler alternative to KL).

    Formula: mean(sum_k r_ik * ||x_i - mu_k||^2)

    This is a simpler loss that encourages points to be close to
    their assigned cluster centers.

    Args:
        cluster_probs: (N, K) soft assignments from ClusterNet
        gmm_params: Current GMM parameters
        data: (N, d) input features

    Returns:
        Scalar loss: weighted squared distances to cluster means
    """
    N, d = data.shape
    K = gmm_params.k

    total_loss = 0.0

    for k in range(K):
        # Squared distance to cluster k
        diff = data - gmm_params.mus[k]  # (N, d)
        squared_dist = torch.sum(diff ** 2, dim=1)  # (N,)

        # Weighted by cluster responsibility
        weighted_dist = cluster_probs[:, k] * squared_dist  # (N,)
        total_loss += weighted_dist.sum()

    # Average over all points
    return total_loss / N
