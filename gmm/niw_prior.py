"""Normal-Inverse-Wishart (NIW) prior for Bayesian GMM."""

import logging
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Module logger
logger = logging.getLogger(__name__)


@dataclass
class NIWPrior:
    """Normal-Inverse-Wishart prior distribution.

    The NIW distribution is a conjugate prior for a multivariate Gaussian
    with unknown mean and covariance. It's parameterized by:
    - m: prior mean (d-dimensional)
    - kappa: prior pseudocount for mean (higher = more peaked around m)
    - psi: prior scale matrix (d x d)
    - nu: prior degrees of freedom (higher = more peaked around psi)
    """

    m: torch.Tensor  # (d,) prior mean
    kappa: float  # prior pseudocount for mean
    psi: torch.Tensor  # (d, d) prior scale matrix
    nu: float  # prior degrees of freedom

    def __post_init__(self):
        """Validate parameters."""
        d = self.m.shape[0]
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.nu < d - 1:
            raise ValueError(f"nu must be >= d-1={d-1}, got {self.nu}")
        if self.psi.shape != (d, d):
            raise ValueError(f"psi shape must be ({d}, {d}), got {self.psi.shape}")

    def to(self, device: torch.device) -> "NIWPrior":
        """Move prior to specified device.

        Args:
            device: Target device

        Returns:
            New NIWPrior instance on target device
        """
        return NIWPrior(
            m=self.m.to(device),
            kappa=self.kappa,
            psi=self.psi.to(device),
            nu=self.nu
        )

    @classmethod
    def from_data(
        cls,
        data: torch.Tensor,
        kappa: float = 0.0001,
        nu_offset: int = 2,
        psi_scale: float = 0.005
    ) -> "NIWPrior":
        """Create NIW prior from data statistics.

        Args:
            data: (N, d) data tensor
            kappa: Prior pseudocount for mean (default: 0.0001, weak prior)
            nu_offset: nu = d + nu_offset (default: 2)
            psi_scale: Scale for psi = I * std(data) * psi_scale

        Returns:
            NIWPrior instance
        """
        d = data.shape[1]
        device = data.device

        # Prior mean: data mean
        m = data.mean(dim=0)

        # Prior degrees of freedom
        nu = d + nu_offset

        # Prior scale matrix: proportional to identity, scaled by data std
        if psi_scale == "auto":
            # Data-dependent scaling
            data_std = data.std(dim=0).mean().item()
            psi = torch.eye(d, device=device) * data_std * 0.0001
        else:
            # Fixed scaling
            psi = torch.eye(d, device=device) * psi_scale

        return cls(m=m, kappa=kappa, psi=psi, nu=nu)

    def compute_posterior(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        cluster_mean: Optional[torch.Tensor] = None
    ) -> "NIWPrior":
        """Compute posterior hyperparameters given data.

        Args:
            data: (N, d) data points
            weights: (N,) soft assignment weights (default: uniform)
            cluster_mean: (d,) mean to use for scatter computation.
                         If None, uses data mean.

        Returns:
            Posterior NIWPrior with updated hyperparameters
        """
        N, d = data.shape
        device = data.device

        # Device consistency checks - move inputs to data's device if needed
        if self.m.device != device:
            # Prior is on different device - this shouldn't happen if caller used .to()
            # but handle gracefully by working on data's device
            logger.warning(f"NIW prior device mismatch: prior on {self.m.device}, data on {device}. Moving prior tensors.")
            m_local = self.m.to(device)
            psi_local = self.psi.to(device)
        else:
            m_local = self.m
            psi_local = self.psi

        if weights is None:
            weights = torch.ones(N, device=device)
        elif weights.device != device:
            logger.debug(f"Moving weights from {weights.device} to {device}")
            weights = weights.to(device)

        if cluster_mean is not None and cluster_mean.device != device:
            logger.debug(f"Moving cluster_mean from {cluster_mean.device} to {device}")
            cluster_mean = cluster_mean.to(device)

        # Effective number of points
        N_eff = weights.sum().item()

        # Data mean (for posterior mean computation)
        if N_eff > 0:
            weighted_sum = (weights.unsqueeze(1) * data).sum(dim=0)
            data_mean = weighted_sum / N_eff
        else:
            data_mean = torch.zeros(d, device=device)

        # Use provided cluster mean or data mean for scatter
        if cluster_mean is None:
            cluster_mean = data_mean

        # Posterior kappa
        kappa_post = self.kappa + N_eff

        # Posterior mean
        # Use m_local for device-safe computation
        m_post = (self.kappa * m_local + weighted_sum if N_eff > 0 else self.kappa * m_local) / kappa_post

        # Posterior nu
        nu_post = self.nu + N_eff

        # Posterior psi
        # psi_star = psi + S + (kappa * N / kappa_star) * (mu - m)(mu - m)^T
        # Where S = (X - mu)^T @ (X - mu) for provided cluster mean mu

        # Compute scatter matrix S = sum_i w_i * (x_i - mu)(x_i - mu)^T
        diff = data - cluster_mean  # (N, d)
        if weights is not None:
            # Weighted scatter
            weighted_diff = weights.unsqueeze(1) * diff  # (N, d)
            S = diff.T @ weighted_diff  # (d, d)
        else:
            S = diff.T @ diff  # (d, d)

        # Deviation term between cluster mean and prior mean
        # Use m_local for device-safe computation
        mean_diff = cluster_mean - m_local
        deviation_term = (self.kappa * N_eff / kappa_post) * torch.outer(mean_diff, mean_diff)

        # Posterior psi (no nu scaling)
        # Use psi_local for device-safe computation
        psi_post = psi_local + S + deviation_term

        return NIWPrior(m=m_post, kappa=kappa_post, psi=psi_post, nu=nu_post)

    def map_estimate(self) -> tuple:
        """Compute MAP estimates of mean and covariance.

        Returns:
            Tuple of (mu, cov) where:
                mu: (d,) MAP estimate of mean
                cov: (d, d) MAP estimate of covariance

        See supplementary material Eqs. 18-19.

        """
        d = self.m.shape[0]

        # MAP mean (Eq. 19)
        mu = self.m

        # MAP covariance
        # For posterior: nu = nu_prior + N_k, so this becomes:
        #   cov = psi_post / (nu_prior + N_k + d + 2)
        cov = self.psi / (self.nu + d + 2)

        # Check for positive definiteness and regularize if needed
        cov = self._stabilize_covariance(cov)

        return mu, cov

    def _stabilize_covariance(self, cov: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
        """Stabilize covariance matrix to ensure numerical stability.

        Args:
            cov: (d, d) covariance matrix
            epsilon: Regularization parameter (default: 1e-5)

        Returns:
            Stabilized covariance matrix
        """
        # Check for NaN values
        if torch.isnan(cov).any():
            # Fallback to scaled identity
            d = cov.shape[0]
            return torch.eye(d, device=cov.device) * 0.005  # prior_sigma_scale

        # Check positive definiteness and add regularization if needed
        try:
            torch.linalg.cholesky(cov)
        except RuntimeError:
            # Not positive definite - add epsilon to diagonal
            d = cov.shape[0]
            cov = cov + epsilon * torch.eye(d, device=cov.device)

        # Also check for infinite or extremely large values
        if torch.isinf(cov).any() or (cov.abs() > 1e10).any():
            d = cov.shape[0]
            return torch.eye(d, device=cov.device) * 0.005

        return cov
