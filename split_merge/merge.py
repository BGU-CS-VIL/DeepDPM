"""Merge proposal and acceptance logic."""

import logging
import torch
import numpy as np
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

# Module logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gmm.parameters import GMMParameters
    from gmm.niw_prior import NIWPrior

from configs.base import MergeConfig

# Module-level RNG for reproducible stochastic acceptance
_merge_rng: Optional[np.random.Generator] = None


def set_merge_seed(seed: int):
    """Set the seed for stochastic merge acceptance.

    Args:
        seed: Random seed for reproducibility
    """
    global _merge_rng
    _merge_rng = np.random.default_rng(seed)


def _get_random_value() -> float:
    """Get a random value from the seeded RNG or fallback to np.random."""
    global _merge_rng
    if _merge_rng is not None:
        return _merge_rng.random()
    else:
        # Fallback if seed not set (for backwards compatibility)
        return np.random.rand()


def propose_merges(
    gmm_params: "GMMParameters",
    data: torch.Tensor,
    cluster_assignments: torch.Tensor,
    prior: "NIWPrior",
    config: MergeConfig
) -> List[Tuple[int, int]]:
    """Propose which cluster pairs to merge.

    For each cluster k:
        1. Find k_nearest neighbor clusters (by distance between means)
        2. For each neighbor k':
            a. Compute LL_k, LL_k' under separate clusters
            b. Compute LL_{k,k'} under merged cluster
            c. Hastings ratio: H = lgamma(N_k + N_k') + LL_{k,k'}
                                 - log(alpha) - lgamma(N_k) - LL_k
                                 - lgamma(N_k') - LL_k'
            d. Accept if H > 0 or with probability min(1, exp(H))

    Args:
        gmm_params: Current GMM parameters
        data: (N, d) input features
        cluster_assignments: (N,) hard cluster assignments
        prior: NIW prior
        config: Merge configuration

    Returns:
        List of (cluster_idx1, cluster_idx2) pairs to merge
        (sorted to avoid conflicts)
    """
    K = gmm_params.k
    if K <= 1:
        return []  # Can't merge if only one cluster

    merges_to_accept = []
    merged_clusters = set()  # Track clusters already involved in merges

    # For each cluster, find nearest neighbors
    for k in range(K):
        if k in merged_clusters:
            continue  # Already part of a merge

        # Find k_nearest neighbors
        neighbors = _find_nearest_neighbors(
            k, gmm_params, config.k_nearest
        )

        # Try merging with each neighbor
        for k_prime in neighbors:
            if k_prime in merged_clusters:
                continue  # Already part of a merge

            # Don't consider merging with self
            if k == k_prime:
                continue

            # Ensure we don't duplicate proposals (k, k') and (k', k)
            if (k_prime, k) in merges_to_accept or (k, k_prime) in merges_to_accept:
                continue

            # Get data for both clusters
            mask_k = cluster_assignments == k
            mask_k_prime = cluster_assignments == k_prime
            mask_merged = mask_k | mask_k_prime

            data_k = data[mask_k]
            data_k_prime = data[mask_k_prime]
            data_merged = data[mask_merged]

            N_k = data_k.shape[0]
            N_k_prime = data_k_prime.shape[0]
            N_merged = data_merged.shape[0]

            if N_k == 0 or N_k_prime == 0:
                continue

            # Compute Hastings ratio
            try:
                # Marginal likelihoods
                posterior_k = prior.compute_posterior(data_k)
                from gmm.marginal_likelihood import marginal_log_likelihood
                ll_k = marginal_log_likelihood(prior, posterior_k, N_k)

                posterior_k_prime = prior.compute_posterior(data_k_prime)
                ll_k_prime = marginal_log_likelihood(prior, posterior_k_prime, N_k_prime)

                posterior_merged = prior.compute_posterior(data_merged)
                ll_merged = marginal_log_likelihood(prior, posterior_merged, N_merged)

                # Log Hastings ratio (inverse of split)
                log_H = math.lgamma(N_merged) + ll_merged - \
                        np.log(config.alpha) - \
                        math.lgamma(N_k) - ll_k - \
                        math.lgamma(N_k_prime) - ll_k_prime

                # Accept/reject decision
                accept = False
                if config.stochastic_accept:
                    # Use numerically stable computation to avoid overflow
                    if log_H > 0:
                        accept_prob = 1.0
                    else:
                        # Cap to avoid underflow (exp(-700) is essentially 0)
                        accept_prob = np.exp(max(log_H, -700))

                    if _get_random_value() < accept_prob:
                        accept = True
                else:
                    if log_H > 0:
                        accept = True

                if accept:
                    # Ensure k < k_prime for consistent ordering
                    if k < k_prime:
                        merges_to_accept.append((k, k_prime))
                    else:
                        merges_to_accept.append((k_prime, k))

                    # Mark both clusters as merged
                    merged_clusters.add(k)
                    merged_clusters.add(k_prime)
                    break  # Don't try other neighbors for this cluster

            except Exception as e:
                # Skip this merge if computation fails
                logger.warning(f"Merge proposal for clusters {k}, {k_prime} failed: {e}")
                continue

    return merges_to_accept


def _find_nearest_neighbors(
    cluster_idx: int,
    gmm_params: "GMMParameters",
    k_nearest: int
) -> List[int]:
    """Find k nearest neighbor clusters by distance between means.

    Args:
        cluster_idx: Index of cluster
        gmm_params: GMM parameters
        k_nearest: Number of nearest neighbors

    Returns:
        List of neighbor cluster indices (sorted by distance)
    """
    K = gmm_params.k
    mu_k = gmm_params.mus[cluster_idx]

    # Compute distances to all other clusters
    distances = []
    for k_prime in range(K):
        if k_prime != cluster_idx:
            mu_k_prime = gmm_params.mus[k_prime]
            dist = torch.norm(mu_k - mu_k_prime).item()
            distances.append((dist, k_prime))

    # Sort by distance and return k_nearest
    distances.sort()
    neighbors = [k_prime for _, k_prime in distances[:k_nearest]]

    return neighbors
