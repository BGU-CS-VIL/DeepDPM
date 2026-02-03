"""Split proposal and acceptance logic for DeepDPM."""

import logging
import torch
import numpy as np
import math
from typing import List, Optional, TYPE_CHECKING

# Module logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gmm.parameters import GMMParameters
    from gmm.niw_prior import NIWPrior

from configs.base import SplitConfig
from gmm.marginal_likelihood import compute_hastings_ratio_split, marginal_log_likelihood

# Module-level RNG for reproducible stochastic acceptance
_split_rng: Optional[np.random.Generator] = None


def set_split_seed(seed: int):
    """Set the seed for stochastic split acceptance.

    Args:
        seed: Random seed for reproducibility
    """
    global _split_rng
    _split_rng = np.random.default_rng(seed)


def _get_random_value() -> float:
    """Get a random value from the seeded RNG or fallback to np.random."""
    global _split_rng
    if _split_rng is not None:
        return _split_rng.random()
    else:
        # Fallback if seed not set (for backwards compatibility)
        return np.random.rand()


def comp_subclusters_params_min_dist(
    data_k: torch.Tensor,
    mu_sub_1: torch.Tensor,
    mu_sub_2: torch.Tensor
) -> torch.Tensor:
    """Compute subcluster assignments by minimum Euclidean distance.

    This is the fallback when SubclusterNet is not available or ignore_subclusters=True.

    Args:
        data_k: (N_k, d) data points in cluster k
        mu_sub_1: (d,) first subcluster mean
        mu_sub_2: (d,) second subcluster mean

    Returns:
        (N_k,) hard assignments: 0 for subcluster 1, 1 for subcluster 2
    """
    dists_0 = torch.sqrt(torch.sum((data_k - mu_sub_1) ** 2, dim=1))
    dists_1 = torch.sqrt(torch.sum((data_k - mu_sub_2) ** 2, dim=1))
    assignments = torch.stack([dists_0, dists_1]).argmin(dim=0)
    return assignments


def propose_splits(
    gmm_params: "GMMParameters",
    data: torch.Tensor,
    subcluster_assignments: torch.Tensor,
    cluster_assignments: torch.Tensor,
    prior: "NIWPrior",
    config: SplitConfig
) -> List[int]:
    """Propose which clusters to split based on Hastings ratio.

    Primary path (ignore_subclusters=False):
        Uses SubclusterNet predictions to partition data into two subclusters.
        The Hastings ratio is computed using the LEARNED subcluster means (mus_sub).

    Fallback path (ignore_subclusters=True):
        Uses minimum Euclidean distance to subcluster means (mus_sub) to partition.
        This is useful when SubclusterNet predictions are unreliable.

    For each cluster k:
        1. Partition data using SubclusterNet or min-distance to mus_sub
        2. Compute marginal log-likelihood LL_k under single cluster
        3. Compute marginal log-likelihoods LL_{k,1}, LL_{k,2} under two subclusters
        4. Compute Hastings ratio: H = log(alpha) + lgamma(N_{k,1}) + LL_{k,1}
                                       + lgamma(N_{k,2}) + LL_{k,2}
                                       - lgamma(N_k) - LL_k
        5. Accept if H > 0 or stochastically with probability min(1, exp(H))

    Args:
        gmm_params: Current GMM parameters (includes mus_sub)
        data: (N, d) input features
        subcluster_assignments: (N, 2*K) soft subcluster assignments from SubclusterNet
        cluster_assignments: (N,) hard cluster assignments
        prior: NIW prior for Bayesian model comparison
        config: Split configuration

    Returns:
        List of cluster indices to split (sorted in reverse order for safe iteration)
    """
    K = gmm_params.k
    splits_to_accept = []

    logger.debug(f"{'='*80}")
    logger.debug(f"SPLIT PROPOSAL: Evaluating {K} clusters")
    logger.debug(f"{'='*80}")

    unique, counts = torch.unique(cluster_assignments, return_counts=True)
    logger.debug(f"Cluster assignment distribution:")
    for cluster_id, count in zip(unique.tolist(), counts.tolist()):
        logger.debug(f"  Cluster {cluster_id}: {count} points")

    for k in range(K):
        # Get points in cluster k
        mask_k = cluster_assignments == k
        if mask_k.sum() < config.min_cluster_size:
            continue  # Skip very small clusters

        data_k = data[mask_k]  # (N_k, d)
        N_k = data_k.shape[0]

        logger.debug(f"Cluster {k}: N={N_k} points")

        if N_k < 5:  # Need enough points for meaningful split
            logger.debug(f"  SKIP: Too few points (N={N_k} < 5)")
            continue

        try:
            # Partition data into two subclusters
            if config.ignore_subclusters:
                # FALLBACK: Use min-distance to subcluster means
                sub_assignments_k = comp_subclusters_params_min_dist(
                    data_k,
                    gmm_params.mus_sub[2 * k],
                    gmm_params.mus_sub[2 * k + 1]
                )
                # sub_assignments_k: (N_k,) with values 0 or 1
                mask_sub1 = sub_assignments_k == 0
                mask_sub2 = sub_assignments_k == 1
            else:
                # PRIMARY: Use SubclusterNet predictions
                # subcluster_assignments is (N, 2*K) with soft probabilities
                # Each cluster k has subclusters at indices 2*k and 2*k+1
                # We only look at the subcluster pair for THIS cluster
                sub_probs_k = subcluster_assignments[mask_k][:, 2*k:2*k+2]  # (N_k, 2)

                # Argmax within this cluster's subcluster pair
                sub_assign_k = sub_probs_k.argmax(dim=1)  # (N_k,) with values 0 or 1

                # Points assigned to subcluster 0 go to partition 1
                # Points assigned to subcluster 1 go to partition 2
                mask_sub1 = sub_assign_k == 0
                mask_sub2 = sub_assign_k == 1

            data_sub1 = data_k[mask_sub1]  # (N_{k,1}, d)
            data_sub2 = data_k[mask_sub2]  # (N_{k,2}, d)

            N_sub1 = data_sub1.shape[0]
            N_sub2 = data_sub2.shape[0]

            logger.debug(f"  Subcluster partition: N_sub1={N_sub1}, N_sub2={N_sub2}")

            # Skip if one subcluster is too small
            if N_sub1 < 5 or N_sub2 < 5:
                logger.debug(f"  SKIP: Subcluster too small (N_sub1={N_sub1}, N_sub2={N_sub2})")
                continue

            # Compute marginal likelihoods
            # IMPORTANT: Use STORED GMM means (mus, mus_sub), not data means!
            # The stored means come from posterior updates with the NIW prior.

            # For the full cluster - use stored cluster mean
            posterior_k = prior.compute_posterior(
                data_k, cluster_mean=gmm_params.mus[k]
            )
            ll_k = marginal_log_likelihood(prior, posterior_k, N_k)

            # For subcluster 1 - use stored subcluster mean
            posterior_sub1 = prior.compute_posterior(
                data_sub1, cluster_mean=gmm_params.mus_sub[2 * k]
            )
            ll_sub1 = marginal_log_likelihood(prior, posterior_sub1, N_sub1)

            # For subcluster 2 - use stored subcluster mean
            posterior_sub2 = prior.compute_posterior(
                data_sub2, cluster_mean=gmm_params.mus_sub[2 * k + 1]
            )
            ll_sub2 = marginal_log_likelihood(prior, posterior_sub2, N_sub2)

            # Log Hastings ratio (Eq. 2 from paper)
            gamma_contrib = math.lgamma(N_sub1) + math.lgamma(N_sub2) - math.lgamma(N_k)
            ll_contrib = (ll_sub1 + ll_sub2) - ll_k
            log_H = np.log(config.alpha) + gamma_contrib + ll_contrib

            logger.debug(f"  Hastings ratio components:")
            logger.debug(f"    log(alpha) = {np.log(config.alpha):.4f}")
            logger.debug(f"    gamma_contrib = lgamma({N_sub1}) + lgamma({N_sub2}) - lgamma({N_k}) = {gamma_contrib:.4f}")
            logger.debug(f"    ll_k = {ll_k:.4f}")
            logger.debug(f"    ll_sub1 = {ll_sub1:.4f}")
            logger.debug(f"    ll_sub2 = {ll_sub2:.4f}")
            logger.debug(f"    ll_contrib = (ll_sub1 + ll_sub2) - ll_k = {ll_contrib:.4f}")
            logger.debug(f"    log_H = {log_H:.4f}")

            # Accept/reject decision
            if config.stochastic_accept:
                # Stochastic acceptance: accept with probability min(1, exp(H))
                if log_H > 0:
                    accept_prob = 1.0
                else:
                    # Cap to avoid underflow (exp(-700) is essentially 0)
                    accept_prob = np.exp(max(log_H, -700))

                rand_val = _get_random_value()
                logger.debug(f"  Stochastic acceptance: p={accept_prob:.6f}, rand={rand_val:.6f}")

                if rand_val < accept_prob:
                    splits_to_accept.append(k)
                    logger.debug(f"  SPLIT ACCEPTED for cluster {k}")
                else:
                    logger.debug(f"  SPLIT REJECTED for cluster {k}")
            else:
                # Deterministic acceptance: accept if H > 0
                if log_H > 0:
                    splits_to_accept.append(k)
                    logger.debug(f"  SPLIT ACCEPTED for cluster {k} (log_H > 0)")
                else:
                    logger.debug(f"  SPLIT REJECTED for cluster {k} (log_H <= 0)")

        except (torch.linalg.LinAlgError, RuntimeError) as e:
            # Handle numerical issues (singular matrices, non-positive definite, etc.)
            error_str = str(e).lower()
            if "singular" in error_str or "positive" in error_str or "cholesky" in error_str:
                logger.warning(f"Numerical issue in split proposal for cluster {k}: {e}")
                continue
            # Re-raise unexpected RuntimeErrors
            raise
        except ValueError as e:
            # Handle value errors (e.g., invalid dimensions, empty clusters)
            logger.warning(f"Value error in split proposal for cluster {k}: {e}")
            continue

    logger.debug(f"{'='*80}")
    logger.debug(f"SPLIT SUMMARY: {len(splits_to_accept)} splits accepted: {splits_to_accept}")
    logger.debug(f"{'='*80}")

    # Return in reverse order so we can iterate safely
    # (splitting changes indices of later clusters)
    return sorted(splits_to_accept, reverse=True)
