"""Marginal likelihood computation for Bayesian model comparison."""

import torch
import numpy as np
import math
from .niw_prior import NIWPrior


def marginal_log_likelihood(
    prior: NIWPrior,
    posterior: NIWPrior,
    n_points: int
) -> float:
    """Compute marginal data log-likelihood p(X | λ).

    This implements Equation 17 from the DeepDPM paper (supplementary material):
        log p(X) = -N*d/2 * log(π)
                   + log(Γ_d(ν*/2)) - log(Γ_d(ν/2))
                   + ν/2 * log(|ν*Ψ|) - ν*/2 * log(|ν*Ψ*|)
                   + d/2 * (log(κ) - log(κ*))

    Args:
        prior: NIW prior hyperparameters (κ, m, ν, Ψ)
        posterior: NIW posterior hyperparameters (κ*, m*, ν*, Ψ*)
        n_points: Number of data points N

    Returns:
        Scalar marginal log-likelihood

    Note:
        This is used in the Hastings ratio for split/merge proposals.
    """
    d = prior.m.shape[0]
    device = prior.m.device

    # Term 1: -N*d/2 * log(π)
    term1 = -n_points * d / 2 * np.log(np.pi)

    # Term 2: log(Γ_d(ν*/2)) - log(Γ_d(ν/2))
    term2 = _log_multivariate_gamma(posterior.nu / 2, d) - \
            _log_multivariate_gamma(prior.nu / 2, d)

    # Term 3: ν/2 * log(|Ψ|)
    psi_prior = prior.psi + torch.eye(d, device=device) * 1e-6  # Regularize
    sign_prior, logdet_prior = torch.linalg.slogdet(psi_prior)
    term3 = prior.nu / 2 * logdet_prior.item()

    # Term 4: -ν*/2 * log(|Ψ*|)
    psi_posterior = posterior.psi + torch.eye(d, device=device) * 1e-6  # Regularize
    sign_posterior, logdet_posterior = torch.linalg.slogdet(psi_posterior)
    term4 = -posterior.nu / 2 * logdet_posterior.item()

    # Term 5: d/2 * (log(κ) - log(κ*))
    term5 = d / 2 * (np.log(prior.kappa) - np.log(posterior.kappa))

    log_likelihood = term1 + term2 + term3 + term4 + term5

    return log_likelihood


def _log_multivariate_gamma(a: float, d: int) -> float:
    """Compute log of multivariate gamma function.

    Γ_d(a) = π^(d(d-1)/4) * ∏_{j=1}^{d} Γ(a + (1-j)/2)

    Args:
        a: Argument
        d: Dimension

    Returns:
        log(Γ_d(a))
    """
    result = d * (d - 1) / 4 * np.log(np.pi)

    for j in range(1, d + 1):
        # Use lgamma directly to avoid overflow in gamma computation
        result += math.lgamma(a + (1 - j) / 2)

    return result


def compute_hastings_ratio_split(
    data_cluster: torch.Tensor,
    data_sub1: torch.Tensor,
    data_sub2: torch.Tensor,
    prior: NIWPrior,
    alpha: float
) -> float:
    """Compute Hastings ratio for split proposal.

    From paper Equation 2:
        H_s = α * Γ(N_{k,1}) * f_x(X_{k,1}; λ) * Γ(N_{k,2}) * f_x(X_{k,2}; λ)
              / (Γ(N_k) * f_x(X_k; λ))

    Args:
        data_cluster: (N_k, d) points in cluster k
        data_sub1: (N_{k,1}, d) points in subcluster 1
        data_sub2: (N_{k,2}, d) points in subcluster 2
        prior: NIW prior
        alpha: Dirichlet process concentration parameter

    Returns:
        log Hastings ratio
    """
    N_k = data_cluster.shape[0]
    N_k1 = data_sub1.shape[0]
    N_k2 = data_sub2.shape[0]

    # Compute marginal likelihoods
    # For cluster k
    if N_k > 0:
        posterior_k = prior.compute_posterior(data_cluster)
        ll_k = marginal_log_likelihood(prior, posterior_k, N_k)
    else:
        ll_k = 0.0

    # For subcluster 1
    if N_k1 > 0:
        posterior_k1 = prior.compute_posterior(data_sub1)
        ll_k1 = marginal_log_likelihood(prior, posterior_k1, N_k1)
    else:
        ll_k1 = 0.0

    # For subcluster 2
    if N_k2 > 0:
        posterior_k2 = prior.compute_posterior(data_sub2)
        ll_k2 = marginal_log_likelihood(prior, posterior_k2, N_k2)
    else:
        ll_k2 = 0.0

    # Log Hastings ratio
    log_H = np.log(alpha) + \
            _log_gamma(N_k1) + ll_k1 + \
            _log_gamma(N_k2) + ll_k2 - \
            _log_gamma(N_k) - ll_k

    return log_H


def compute_hastings_ratio_merge(
    data_cluster1: torch.Tensor,
    data_cluster2: torch.Tensor,
    data_merged: torch.Tensor,
    prior: NIWPrior,
    alpha: float
) -> float:
    """Compute Hastings ratio for merge proposal.

    H_m = 1 / H_s (from split formula)

    Args:
        data_cluster1: (N_{k1}, d) points in cluster k1
        data_cluster2: (N_{k2}, d) points in cluster k2
        data_merged: (N_{k1} + N_{k2}, d) points in merged cluster
        prior: NIW prior
        alpha: Dirichlet process concentration parameter

    Returns:
        log Hastings ratio
    """
    N_k1 = data_cluster1.shape[0]
    N_k2 = data_cluster2.shape[0]
    N_merged = data_merged.shape[0]

    # Compute marginal likelihoods
    if N_k1 > 0:
        posterior_k1 = prior.compute_posterior(data_cluster1)
        ll_k1 = marginal_log_likelihood(prior, posterior_k1, N_k1)
    else:
        ll_k1 = 0.0

    if N_k2 > 0:
        posterior_k2 = prior.compute_posterior(data_cluster2)
        ll_k2 = marginal_log_likelihood(prior, posterior_k2, N_k2)
    else:
        ll_k2 = 0.0

    if N_merged > 0:
        posterior_merged = prior.compute_posterior(data_merged)
        ll_merged = marginal_log_likelihood(prior, posterior_merged, N_merged)
    else:
        ll_merged = 0.0

    # Log Hastings ratio (inverse of split)
    log_H = _log_gamma(N_merged) + ll_merged - \
            np.log(alpha) - \
            _log_gamma(N_k1) - ll_k1 - \
            _log_gamma(N_k2) - ll_k2

    return log_H


def _log_gamma(n: int) -> float:
    """Compute log(Γ(n)) = log((n-1)!) for integer n.

    Args:
        n: Non-negative integer

    Returns:
        log(Γ(n))
    """
    if n <= 0:
        return 0.0
    return math.lgamma(n)
