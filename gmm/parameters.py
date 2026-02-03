"""GMM Parameters container and operations."""

import logging
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .niw_prior import NIWPrior
from kmeans_pytorch import kmeans as gpu_kmeans

# Module logger
logger = logging.getLogger(__name__)

# Numerical stability constants
EPS = 1e-10  # Epsilon for log/division operations
MIN_PI_THRESHOLD = 1e-8  # Minimum mixture weight before normalization warning

# Module-level seed for reproducible K-means initialization
_kmeans_seed: int | None = None
_kmeans_call_counter: int = 0


def set_kmeans_seed(seed: int | None) -> None:
    """Set the seed for K-means initialization.

    Args:
        seed: Random seed, or None to disable seeding
    """
    global _kmeans_seed, _kmeans_call_counter
    _kmeans_seed = seed
    _kmeans_call_counter = 0


def _seed_before_kmeans() -> None:
    """Seed PyTorch RNG before a K-means call for reproducibility."""
    global _kmeans_call_counter
    if _kmeans_seed is not None:
        # Use seed + counter to ensure different calls get different but deterministic seeds
        torch.manual_seed(_kmeans_seed + _kmeans_call_counter)
        _kmeans_call_counter += 1


@dataclass
class GMMParameters:
    """Container for Gaussian Mixture Model parameters.

    Maintains parameters for K clusters and 2*K subclusters:
    - Cluster k has parameters: mus[k], covs[k], pi[k]
    - Cluster k's subclusters have parameters:
        - Subcluster 0: mus_sub[2k], covs_sub[2k], pi_sub[2k]
        - Subcluster 1: mus_sub[2k+1], covs_sub[2k+1], pi_sub[2k+1]
    """

    mus: torch.Tensor  # (K, d) cluster means
    covs: torch.Tensor  # (K, d, d) cluster covariances
    pi: torch.Tensor  # (K,) cluster mixture weights

    mus_sub: torch.Tensor  # (2*K, d) subcluster means
    covs_sub: torch.Tensor  # (2*K, d, d) subcluster covariances
    pi_sub: torch.Tensor  # (2*K,) subcluster weights

    @property
    def k(self) -> int:
        """Current number of clusters."""
        return self.mus.shape[0]

    @property
    def d(self) -> int:
        """Data dimensionality."""
        return self.mus.shape[1]

    @classmethod
    def initialize_from_kmeans(
        cls,
        data: torch.Tensor,
        k: int,
        prior: NIWPrior
    ) -> "GMMParameters":
        """Initialize parameters using K-means clustering.

        Args:
            data: (N, d) input features
            k: Number of initial clusters
            prior: NIW prior for MAP estimation

        Returns:
            Initialized GMMParameters
        """
        N, d = data.shape
        device = data.device

        # Run GPU K-means
        if k == 1:
            # Single cluster uses sklearn
            data_np = data.cpu().numpy()
            kmeans_sklearn = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans_sklearn.fit_predict(data_np)
            labels = torch.from_numpy(labels)
            mus_kmeans = torch.from_numpy(kmeans_sklearn.cluster_centers_)
        else:
            # Use GPU K-means for k > 1
            _seed_before_kmeans()
            labels, mus_kmeans = gpu_kmeans(
                X=data.detach(),
                num_clusters=k,
                device=device
            )
            # Move results to CPU for GMM computations
            labels = labels.cpu()
            mus_kmeans = mus_kmeans.cpu()

        # GMM computations on CPU
        data_cpu = data.cpu()

        # Move prior to CPU for GMM computations
        prior_cpu = prior.to(torch.device('cpu'))

        # Initialize cluster parameters using K-means centers (on CPU)
        mus = torch.zeros(k, d)
        covs = torch.zeros(k, d, d)
        pi = torch.zeros(k)

        for cluster_idx in range(k):
            mask = labels == cluster_idx
            cluster_data = data_cpu[mask]

            if cluster_data.shape[0] > 0:
                # Compute posterior MAP estimate using K-means center
                N_k = cluster_data.shape[0]
                # Pass K-means center for scatter computation
                posterior = prior_cpu.compute_posterior(
                    cluster_data,
                    cluster_mean=mus_kmeans[cluster_idx]
                )
                mu, cov = posterior.map_estimate()
                mus[cluster_idx] = mu
                covs[cluster_idx] = cov
                pi[cluster_idx] = mask.sum().float() / N
            else:
                # Empty cluster - use prior
                mu, cov = prior_cpu.map_estimate()
                mus[cluster_idx] = mu
                covs[cluster_idx] = cov
                pi[cluster_idx] = 1.0 / k

        # Normalize pi
        pi = pi / pi.sum()

        # Initialize subclusters (2 per cluster)
        mus_sub = torch.zeros(2 * k, d)
        covs_sub = torch.zeros(2 * k, d, d)
        pi_sub = torch.zeros(2 * k)

        for cluster_idx in range(k):
            # Run GPU K-means with k=2 on each cluster
            mask = labels == cluster_idx
            cluster_data_gpu = data[mask]  # Keep on GPU for K-means
            cluster_data_cpu = data_cpu[mask]  # CPU version for posterior

            if cluster_data_gpu.shape[0] >= 2:
                # GPU K-means with k=2
                _seed_before_kmeans()
                sublabels, mus_sub_kmeans = gpu_kmeans(
                    X=cluster_data_gpu.detach(),
                    num_clusters=2,
                    device=device
                )
                # Move results to CPU
                sublabels = sublabels.cpu()
                mus_sub_kmeans = mus_sub_kmeans.cpu()

                for sub_idx in range(2):
                    submask = sublabels == sub_idx
                    subcluster_data = cluster_data_cpu[submask]

                    if subcluster_data.shape[0] > 0:
                        N_sub = subcluster_data.shape[0]
                        # Pass K-means center for scatter computation
                        posterior = prior_cpu.compute_posterior(
                            subcluster_data,
                            cluster_mean=mus_sub_kmeans[sub_idx]
                        )
                        mu_sub, cov_sub = posterior.map_estimate()
                        mus_sub[2 * cluster_idx + sub_idx] = mu_sub
                        covs_sub[2 * cluster_idx + sub_idx] = cov_sub
                        pi_sub[2 * cluster_idx + sub_idx] = \
                            submask.sum().float() / mask.sum().float()
                    else:
                        # Use cluster parameters
                        mus_sub[2 * cluster_idx + sub_idx] = mus[cluster_idx]
                        covs_sub[2 * cluster_idx + sub_idx] = covs[cluster_idx]
                        pi_sub[2 * cluster_idx + sub_idx] = 0.5
            else:
                # Not enough points - duplicate cluster parameters
                for sub_idx in range(2):
                    mus_sub[2 * cluster_idx + sub_idx] = mus[cluster_idx]
                    covs_sub[2 * cluster_idx + sub_idx] = covs[cluster_idx]
                    pi_sub[2 * cluster_idx + sub_idx] = 0.5

        # Normalize subcluster weights (each pair should sum to parent pi)
        for cluster_idx in range(k):
            sub_sum = pi_sub[2*cluster_idx] + pi_sub[2*cluster_idx + 1]
            if sub_sum > 0:
                pi_sub[2*cluster_idx] = pi_sub[2*cluster_idx] / sub_sum * pi[cluster_idx]
                pi_sub[2*cluster_idx + 1] = pi_sub[2*cluster_idx + 1] / sub_sum * pi[cluster_idx]

        # Move all parameters to target device before returning
        return cls(
            mus=mus.to(device),
            covs=covs.to(device),
            pi=pi.to(device),
            mus_sub=mus_sub.to(device),
            covs_sub=covs_sub.to(device),
            pi_sub=pi_sub.to(device)
        )

    def compute_log_responsibilities(self, data: torch.Tensor) -> torch.Tensor:
        """Compute E-step log-responsibilities: log p(z_i=k | x_i, params).

        Args:
            data: (N, d) input features

        Returns:
            (N, K) log cluster responsibilities (unnormalized)

        Implementation uses numerically stable log-sum-exp trick.
        """
        N = data.shape[0]
        K = self.k
        device = data.device

        log_resp = torch.zeros(N, K, device=device)

        for k in range(K):
            # Multivariate normal log-likelihood
            log_resp[:, k] = self._mvn_log_prob(
                data, self.mus[k], self.covs[k]
            ) + torch.log(self.pi[k] + EPS)

        return log_resp

    def compute_responsibilities(self, data: torch.Tensor) -> torch.Tensor:
        """Compute normalized E-step responsibilities.

        Args:
            data: (N, d) input features

        Returns:
            (N, K) cluster responsibilities (sum to 1 over K)
        """
        log_resp = self.compute_log_responsibilities(data)

        # Normalize using log-sum-exp trick
        log_resp_normalized = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp_normalized)

    def update_from_responsibilities(
        self,
        data: torch.Tensor,
        responsibilities: torch.Tensor,
        prior: NIWPrior
    ):
        """M-step: Update parameters using weighted MAP estimates.

        Args:
            data: (N, d) input features
            responsibilities: (N, K) soft cluster assignments
            prior: NIW prior for MAP estimation

        Updates self.mus, self.covs, self.pi in-place.
        """
        N, d = data.shape
        K = self.k
        device = data.device

        # Validate responsibilities for NaN/Inf
        if torch.isnan(responsibilities).any() or torch.isinf(responsibilities).any():
            logger.warning("NaN/Inf detected in responsibilities, skipping GMM update")
            return

        # Move to CPU for posterior computations
        data_cpu = data.cpu()
        responsibilities_cpu = responsibilities.cpu()
        prior_cpu = prior.to(torch.device('cpu'))

        for k in range(K):
            weights = responsibilities_cpu[:, k]

            # Update cluster parameters
            N_k = weights.sum().item()
            posterior = prior_cpu.compute_posterior(data_cpu, weights)
            mu, cov = posterior.map_estimate()

            self.mus[k] = mu.to(device)
            self.covs[k] = cov.to(device)
            self.pi[k] = weights.sum() / N

        # Normalize pi with numerical guard
        pi_sum = self.pi.sum()
        if pi_sum < EPS:
            logger.warning(f"pi_sum={pi_sum:.2e} is very small, using uniform weights")
            self.pi = torch.ones_like(self.pi) / K
        else:
            self.pi = self.pi / pi_sum

    def update_subclusters_from_assignments(
        self,
        data: torch.Tensor,
        cluster_assignments: torch.Tensor,
        subcluster_responsibilities: torch.Tensor,
        prior: NIWPrior
    ):
        """Update subcluster parameters from soft subcluster assignments.

        Includes degeneracy protection - if a subcluster gets near-zero weight,
        re-initialize both subclusters of that cluster using K-means.

        Args:
            data: (N, d) input features
            cluster_assignments: (N,) hard cluster assignments
            subcluster_responsibilities: (N, 2*K) soft subcluster assignments
            prior: NIW prior
        """
        N, d = data.shape
        K = self.k
        device = data.device

        # Move to CPU for posterior computations
        data_cpu = data.cpu()
        cluster_assignments_cpu = cluster_assignments.cpu()
        subcluster_responsibilities_cpu = subcluster_responsibilities.cpu()
        prior_cpu = prior.to(torch.device('cpu'))

        # Minimum weight threshold for degeneracy detection
        # Use relative threshold based on cluster size for better scaling
        MIN_WEIGHT_THRESHOLD = max(1e-6, 1.0 / N)

        for k in range(K):
            mask = cluster_assignments_cpu == k
            N_k = mask.sum().item()

            if N_k == 0:
                continue

            cluster_data = data_cpu[mask]

            # Extract relevant subclusters [2k, 2k+1] for cluster k
            # Input shape: (N, 2*K), extract (N_k, 2)
            cluster_sub_resp = subcluster_responsibilities_cpu[mask, 2*k:2*k+2]  # (N_k, 2)

            # Compute total weights for each subcluster
            weight_sub0 = cluster_sub_resp[:, 0].sum().item()
            weight_sub1 = cluster_sub_resp[:, 1].sum().item()

            # Check for degeneracy: if either subcluster has near-zero weight OR
            # hard assignments don't produce both subclusters
            hard_assignments = cluster_sub_resp.argmax(dim=1)
            n_unique_subclusters = len(torch.unique(hard_assignments))

            is_degenerate = (
                N_k < 2 or
                weight_sub0 < MIN_WEIGHT_THRESHOLD or
                weight_sub1 < MIN_WEIGHT_THRESHOLD or
                n_unique_subclusters < 2
            )

            if is_degenerate:
                # Re-initialize subclusters with 1D K-means
                logger.debug(f"[DEGENERACY] Cluster {k}: N_k={N_k}, weights=({weight_sub0:.1f}, {weight_sub1:.1f}), unique_hard={n_unique_subclusters}")

                if N_k >= 2:
                    # 1D K-means: PCA to 1D, K-means, inverse transform
                    cluster_data_cpu = cluster_data.cpu().numpy()
                    pca = PCA(n_components=1)
                    pca_data = pca.fit_transform(cluster_data_cpu)
                    pca_tensor = torch.from_numpy(pca_data).to(device)
                    _seed_before_kmeans()
                    sub_labels, cluster_centers_1d = gpu_kmeans(
                        X=pca_tensor,
                        num_clusters=2,
                        device=device
                    )

                    # Inverse transform cluster centers to original space
                    mus_sub_init = torch.from_numpy(
                        pca.inverse_transform(cluster_centers_1d.cpu().numpy())
                    ).float().to(device)

                    n_sub0 = (sub_labels == 0).sum().item()
                    n_sub1 = (sub_labels == 1).sum().item()
                    logger.debug(f"  1D K-means split: {n_sub0}/{n_sub1}")

                    # Reinitialize both subclusters using 1D K-means results
                    for sub_idx in range(2):
                        sub_mask = sub_labels == sub_idx
                        count = sub_mask.sum().item()

                        if count > 0:
                            # Use hard weights from K-means labels
                            weights = sub_mask.float()

                            # Compute posterior with 1D K-means center as cluster_mean
                            # This uses the K-means center for scatter computation
                            posterior = prior_cpu.compute_posterior(
                                cluster_data,
                                weights,
                                cluster_mean=mus_sub_init[sub_idx].cpu()
                            )

                            # Get MAP estimate from posterior
                            mu_sub, cov_sub = posterior.map_estimate()
                        else:
                            # Fallback to prior if K-means fails
                            mu_sub, cov_sub = prior_cpu.map_estimate()

                        self.mus_sub[2*k + sub_idx] = mu_sub.to(device)
                        self.covs_sub[2*k + sub_idx] = cov_sub.to(device)
                        self.pi_sub[2*k + sub_idx] = count / N

                else:
                    # Too few points - use prior
                    for sub_idx in range(2):
                        mu_sub, cov_sub = prior_cpu.map_estimate()
                        self.mus_sub[2*k + sub_idx] = mu_sub.to(device)
                        self.covs_sub[2*k + sub_idx] = cov_sub.to(device)
                        self.pi_sub[2*k + sub_idx] = 0.5 / N * self.pi[k]
            else:
                # Normal update using soft assignments
                for sub_idx in range(2):
                    weights = cluster_sub_resp[:, sub_idx]

                    # Update subcluster parameters
                    N_sub = weights.sum().item()
                    posterior = prior_cpu.compute_posterior(cluster_data, weights)
                    mu_sub, cov_sub = posterior.map_estimate()

                    self.mus_sub[2*k + sub_idx] = mu_sub.to(device)
                    self.covs_sub[2*k + sub_idx] = cov_sub.to(device)
                    self.pi_sub[2*k + sub_idx] = weights.sum() / N

        # Normalize subcluster weights (each pair should sum to parent pi)
        for k in range(K):
            sub_sum = self.pi_sub[2*k] + self.pi_sub[2*k + 1]
            if sub_sum > EPS:
                self.pi_sub[2*k] = self.pi_sub[2*k] / sub_sum * self.pi[k]
                self.pi_sub[2*k + 1] = self.pi_sub[2*k + 1] / sub_sum * self.pi[k]
            else:
                # Fallback: equal split if sum is too small
                self.pi_sub[2*k] = 0.5 * self.pi[k]
                self.pi_sub[2*k + 1] = 0.5 * self.pi[k]

    def split_cluster(
        self,
        cluster_idx: int,
        data: torch.Tensor,
        subcluster_assignments: torch.Tensor,
        prior: NIWPrior,
        ignore_subclusters: bool = False
    ):
        """Update parameters after cluster split.

        The paper approach:
        1. NEW CLUSTER MEANS = OLD SUBCLUSTER MEANS (promoted)
           - mus_new[cluster_idx] = mus_sub[2*cluster_idx]
           - mus_new[cluster_idx+1] = mus_sub[2*cluster_idx+1]

        2. NEW SUBCLUSTERS = K-means on SubclusterNet-assigned points
           - For each new cluster, extract points assigned to it
           - Run K-means with k=2 to get new subcluster means

        Args:
            cluster_idx: Index of cluster to split
            data: (N, d) all data points
            subcluster_assignments: (N, 2*K) soft subcluster assignments
            prior: NIW prior
            ignore_subclusters: If True, use min-distance instead of SubclusterNet
        """
        if cluster_idx < 0 or cluster_idx >= self.k:
            raise ValueError(f"Invalid cluster_idx {cluster_idx}, K={self.k}")

        K = self.k
        d = self.d
        device = self.mus.device
        N = data.shape[0]

        # Move prior to CPU for posterior computations
        prior_cpu = prior.to(torch.device('cpu'))

        # Get cluster assignments from current GMM
        with torch.no_grad():
            responsibilities = self.compute_responsibilities(data)
            cluster_assignments = responsibilities.argmax(dim=1)

        # Get data from this cluster
        cluster_mask = cluster_assignments == cluster_idx
        data_k = data[cluster_mask]
        N_k = data_k.shape[0]

        # =====================================================================
        # STEP 1: PROMOTE SUBCLUSTERS TO CLUSTERS
        # New cluster means = old subcluster means
        # =====================================================================
        mu1 = self.mus_sub[2 * cluster_idx].clone()
        cov1 = self.covs_sub[2 * cluster_idx].clone()
        mu2 = self.mus_sub[2 * cluster_idx + 1].clone()
        cov2 = self.covs_sub[2 * cluster_idx + 1].clone()
        pi1 = self.pi_sub[2 * cluster_idx].clone()
        pi2 = self.pi_sub[2 * cluster_idx + 1].clone()

        # Normalize pis (should sum to old cluster's pi)
        total_pi = pi1 + pi2
        if total_pi > 0:
            pi1 = pi1 / total_pi * self.pi[cluster_idx]
            pi2 = pi2 / total_pi * self.pi[cluster_idx]
        else:
            pi1 = 0.5 * self.pi[cluster_idx]
            pi2 = 0.5 * self.pi[cluster_idx]

        # =====================================================================
        # STEP 2: PARTITION DATA USING SUBCLUSTERNET OR MIN-DISTANCE
        # This determines which points go to which new cluster
        # =====================================================================
        if N_k >= 2:
            if ignore_subclusters:
                # FALLBACK: Use min-distance to subcluster means
                dists_0 = torch.sqrt(torch.sum((data_k - mu1) ** 2, dim=1))
                dists_1 = torch.sqrt(torch.sum((data_k - mu2) ** 2, dim=1))
                partition_labels = torch.stack([dists_0, dists_1]).argmin(dim=0)
            else:
                # PRIMARY: Use SubclusterNet predictions
                # Only look at this cluster's subcluster pair (indices 2*k and 2*k+1)
                sub_probs_k = subcluster_assignments[cluster_mask][:, 2*cluster_idx:2*cluster_idx+2]  # (N_k, 2)

                # Argmax within this cluster's subcluster pair
                partition_labels = sub_probs_k.argmax(dim=1)  # (N_k,) with values 0 or 1

            mask1 = partition_labels == 0
            mask2 = partition_labels == 1

            # Update pis based on actual point counts
            N_k1 = mask1.sum().item()
            N_k2 = mask2.sum().item()
            if N_k1 + N_k2 > 0:
                pi1 = (N_k1 / N) * (N / (N_k1 + N_k2)) * self.pi[cluster_idx]
                pi2 = (N_k2 / N) * (N / (N_k1 + N_k2)) * self.pi[cluster_idx]
        else:
            mask1 = torch.ones(N_k, dtype=torch.bool, device=device)
            mask2 = torch.zeros(N_k, dtype=torch.bool, device=device)

        # =====================================================================
        # STEP 3: CREATE NEW CLUSTER PARAMETER TENSORS (K -> K+1)
        # =====================================================================
        new_mus = torch.zeros(K + 1, d, device=device)
        new_covs = torch.zeros(K + 1, d, d, device=device)
        new_pi = torch.zeros(K + 1, device=device)

        # Copy before split
        new_mus[:cluster_idx] = self.mus[:cluster_idx]
        new_covs[:cluster_idx] = self.covs[:cluster_idx]
        new_pi[:cluster_idx] = self.pi[:cluster_idx]

        # Insert new clusters (promoted subclusters)
        new_mus[cluster_idx] = mu1
        new_mus[cluster_idx + 1] = mu2
        new_covs[cluster_idx] = cov1
        new_covs[cluster_idx + 1] = cov2
        new_pi[cluster_idx] = pi1
        new_pi[cluster_idx + 1] = pi2

        # Copy after split
        new_mus[cluster_idx + 2:] = self.mus[cluster_idx + 1:]
        new_covs[cluster_idx + 2:] = self.covs[cluster_idx + 1:]
        new_pi[cluster_idx + 2:] = self.pi[cluster_idx + 1:]

        # Update cluster parameters
        self.mus = new_mus
        self.covs = new_covs
        self.pi = new_pi / new_pi.sum()

        # =====================================================================
        # STEP 4: CREATE NEW SUBCLUSTERS VIA K-MEANS (following paper)
        # For each new cluster, run K-means with k=2 on its assigned points
        # =====================================================================
        new_mus_sub = torch.zeros(2 * (K + 1), d, device=device)
        new_covs_sub = torch.zeros(2 * (K + 1), d, d, device=device)
        new_pi_sub = torch.zeros(2 * (K + 1), device=device)

        # Copy subclusters before split
        new_mus_sub[:2*cluster_idx] = self.mus_sub[:2*cluster_idx]
        new_covs_sub[:2*cluster_idx] = self.covs_sub[:2*cluster_idx]
        new_pi_sub[:2*cluster_idx] = self.pi_sub[:2*cluster_idx]

        # Initialize subclusters for new cluster 0 (at cluster_idx)
        # Use K-means to find distinct subcluster means
        cluster_0_data = data_k[mask1]
        if cluster_0_data.shape[0] >= 2:
            # GPU K-means for subclusters
            _seed_before_kmeans()
            sublabels, mus_sub_kmeans = gpu_kmeans(
                X=cluster_0_data.detach(),
                num_clusters=2,
                device=device
            )
            # Move results to CPU
            sublabels = sublabels.cpu()
            mus_sub_kmeans = mus_sub_kmeans.cpu()
            cluster_0_data_cpu = cluster_0_data.cpu()

            for sub_idx in range(2):
                submask = sublabels == sub_idx
                subcluster_data = cluster_0_data_cpu[submask]

                if subcluster_data.shape[0] > 0:
                    N_sub = subcluster_data.shape[0]
                    # Pass K-means center for scatter computation
                    posterior = prior_cpu.compute_posterior(
                        subcluster_data,
                        cluster_mean=mus_sub_kmeans[sub_idx]
                    )
                    mu_sub, cov_sub = posterior.map_estimate()
                    new_mus_sub[2*cluster_idx + sub_idx] = mu_sub.to(device)
                    new_covs_sub[2*cluster_idx + sub_idx] = cov_sub.to(device)
                    new_pi_sub[2*cluster_idx + sub_idx] = (
                        submask.sum().float() / cluster_0_data.shape[0] * pi1
                    )
                else:
                    new_mus_sub[2*cluster_idx + sub_idx] = mu1
                    new_covs_sub[2*cluster_idx + sub_idx] = cov1
                    new_pi_sub[2*cluster_idx + sub_idx] = pi1 / 2
        elif cluster_0_data.shape[0] == 1:
            # Single point: use PCA direction from parent cluster for separation
            new_mus_sub[2*cluster_idx] = mu1
            new_mus_sub[2*cluster_idx + 1] = mu1
            new_covs_sub[2*cluster_idx] = cov1
            new_covs_sub[2*cluster_idx + 1] = cov1
            new_pi_sub[2*cluster_idx] = pi1 / 2
            new_pi_sub[2*cluster_idx + 1] = pi1 / 2
        else:
            # No data assigned - use parent cluster's subcluster means as starting point
            # This maintains continuity from before the split
            new_mus_sub[2*cluster_idx] = mu1
            new_mus_sub[2*cluster_idx + 1] = mu1
            new_covs_sub[2*cluster_idx] = cov1
            new_covs_sub[2*cluster_idx + 1] = cov1
            new_pi_sub[2*cluster_idx] = pi1 / 2
            new_pi_sub[2*cluster_idx + 1] = pi1 / 2

        # Initialize subclusters for new cluster 1 (at cluster_idx + 1)
        cluster_1_data = data_k[mask2]
        if cluster_1_data.shape[0] >= 2:
            # GPU K-means for subclusters
            _seed_before_kmeans()
            sublabels, mus_sub_kmeans = gpu_kmeans(
                X=cluster_1_data.detach(),
                num_clusters=2,
                device=device
            )
            # Move results to CPU
            sublabels = sublabels.cpu()
            mus_sub_kmeans = mus_sub_kmeans.cpu()
            cluster_1_data_cpu = cluster_1_data.cpu()

            for sub_idx in range(2):
                submask = sublabels == sub_idx
                subcluster_data = cluster_1_data_cpu[submask]

                if subcluster_data.shape[0] > 0:
                    N_sub = subcluster_data.shape[0]
                    # Pass K-means center for scatter computation
                    posterior = prior_cpu.compute_posterior(
                        subcluster_data,
                        cluster_mean=mus_sub_kmeans[sub_idx]
                    )
                    mu_sub, cov_sub = posterior.map_estimate()
                    new_mus_sub[2*(cluster_idx + 1) + sub_idx] = mu_sub.to(device)
                    new_covs_sub[2*(cluster_idx + 1) + sub_idx] = cov_sub.to(device)
                    new_pi_sub[2*(cluster_idx + 1) + sub_idx] = (
                        submask.sum().float() / cluster_1_data.shape[0] * pi2
                    )
                else:
                    new_mus_sub[2*(cluster_idx + 1) + sub_idx] = mu2
                    new_covs_sub[2*(cluster_idx + 1) + sub_idx] = cov2
                    new_pi_sub[2*(cluster_idx + 1) + sub_idx] = pi2 / 2
        elif cluster_1_data.shape[0] == 1:
            # Single point: use parent mean (same logic as cluster 0)
            new_mus_sub[2*(cluster_idx + 1)] = mu2
            new_mus_sub[2*(cluster_idx + 1) + 1] = mu2
            new_covs_sub[2*(cluster_idx + 1)] = cov2
            new_covs_sub[2*(cluster_idx + 1) + 1] = cov2
            new_pi_sub[2*(cluster_idx + 1)] = pi2 / 2
            new_pi_sub[2*(cluster_idx + 1) + 1] = pi2 / 2
        else:
            # No data assigned - use parent cluster mean
            new_mus_sub[2*(cluster_idx + 1)] = mu2
            new_mus_sub[2*(cluster_idx + 1) + 1] = mu2
            new_covs_sub[2*(cluster_idx + 1)] = cov2
            new_covs_sub[2*(cluster_idx + 1) + 1] = cov2
            new_pi_sub[2*(cluster_idx + 1)] = pi2 / 2
            new_pi_sub[2*(cluster_idx + 1) + 1] = pi2 / 2

        # Copy subclusters after split
        new_mus_sub[2*(cluster_idx + 2):] = self.mus_sub[2*(cluster_idx + 1):]
        new_covs_sub[2*(cluster_idx + 2):] = self.covs_sub[2*(cluster_idx + 1):]
        new_pi_sub[2*(cluster_idx + 2):] = self.pi_sub[2*(cluster_idx + 1):]

        self.mus_sub = new_mus_sub
        self.covs_sub = new_covs_sub
        self.pi_sub = new_pi_sub

    def merge_clusters(
        self,
        cluster_idx1: int,
        cluster_idx2: int,
        data: torch.Tensor,
        prior: NIWPrior
    ):
        """Update parameters after cluster merge.

        Args:
            cluster_idx1: First cluster index
            cluster_idx2: Second cluster index
            data: (N, d) all data points
            prior: NIW prior
        """
        # Ensure idx1 < idx2
        if cluster_idx1 > cluster_idx2:
            cluster_idx1, cluster_idx2 = cluster_idx2, cluster_idx1

        K = self.k
        # Move prior to CPU for posterior computations
        prior_cpu = prior.to(torch.device('cpu'))
        d = self.d
        device = self.mus.device

        # Save old means for computing merged cluster data before updating
        old_mus = self.mus.clone()

        # Compute merged cluster parameters (weighted by pi)
        total_weight = self.pi[cluster_idx1] + self.pi[cluster_idx2]
        w1 = self.pi[cluster_idx1] / total_weight
        w2 = self.pi[cluster_idx2] / total_weight

        merged_mu = w1 * self.mus[cluster_idx1] + w2 * self.mus[cluster_idx2]
        merged_cov = w1 * self.covs[cluster_idx1] + w2 * self.covs[cluster_idx2]
        merged_pi = total_weight

        # Create new parameter tensors (K -> K-1)
        new_mus = torch.zeros(K - 1, d, device=device)
        new_covs = torch.zeros(K - 1, d, d, device=device)
        new_pi = torch.zeros(K - 1, device=device)

        # Before first cluster
        new_mus[:cluster_idx1] = self.mus[:cluster_idx1]
        new_covs[:cluster_idx1] = self.covs[:cluster_idx1]
        new_pi[:cluster_idx1] = self.pi[:cluster_idx1]

        # Merged cluster
        new_mus[cluster_idx1] = merged_mu
        new_covs[cluster_idx1] = merged_cov
        new_pi[cluster_idx1] = merged_pi

        # Between clusters
        new_mus[cluster_idx1 + 1:cluster_idx2] = self.mus[cluster_idx1 + 1:cluster_idx2]
        new_covs[cluster_idx1 + 1:cluster_idx2] = self.covs[cluster_idx1 + 1:cluster_idx2]
        new_pi[cluster_idx1 + 1:cluster_idx2] = self.pi[cluster_idx1 + 1:cluster_idx2]

        # After second cluster
        new_mus[cluster_idx2:] = self.mus[cluster_idx2 + 1:]
        new_covs[cluster_idx2:] = self.covs[cluster_idx2 + 1:]
        new_pi[cluster_idx2:] = self.pi[cluster_idx2 + 1:]

        self.mus = new_mus
        self.covs = new_covs
        self.pi = new_pi / new_pi.sum()

        # Update subclusters (similar logic)
        new_mus_sub = torch.zeros(2 * (K - 1), d, device=device)
        new_covs_sub = torch.zeros(2 * (K - 1), d, d, device=device)
        new_pi_sub = torch.zeros(2 * (K - 1), device=device)

        # Before first
        new_mus_sub[:2*cluster_idx1] = self.mus_sub[:2*cluster_idx1]
        new_covs_sub[:2*cluster_idx1] = self.covs_sub[:2*cluster_idx1]
        new_pi_sub[:2*cluster_idx1] = self.pi_sub[:2*cluster_idx1]

        # Merged cluster: initialize subclusters using GPU K-means
        # Find data points that belong to the merged cluster
        # Use distance to OLD cluster means (before merge) to assign points
        dist_to_idx1 = torch.cdist(data, old_mus[cluster_idx1:cluster_idx1+1]).squeeze(1)
        dist_to_idx2 = torch.cdist(data, old_mus[cluster_idx2:cluster_idx2+1]).squeeze(1)

        # Points closest to either of the two merging clusters
        min_dist_to_merged = torch.minimum(dist_to_idx1, dist_to_idx2)

        # Compute distances to all other clusters
        other_indices = [i for i in range(K) if i not in [cluster_idx1, cluster_idx2]]
        if len(other_indices) > 0:
            other_mus = old_mus[other_indices]
            dist_to_others = torch.cdist(data, other_mus).min(dim=1).values
            merged_mask = min_dist_to_merged < dist_to_others
        else:
            # No other clusters, all data belongs to merged cluster
            merged_mask = torch.ones(data.shape[0], dtype=torch.bool, device=device)

        merged_data = data[merged_mask]

        if merged_data.shape[0] >= 2:
            # GPU K-means for merged subclusters
            _seed_before_kmeans()
            sublabels, mus_sub_kmeans = gpu_kmeans(
                X=merged_data.detach(),
                num_clusters=2,
                device=device
            )
            # Move results to CPU
            sublabels = sublabels.cpu()
            mus_sub_kmeans = mus_sub_kmeans.cpu()
            merged_data_cpu = merged_data.cpu()

            for sub_idx in range(2):
                submask = sublabels == sub_idx
                subcluster_data = merged_data_cpu[submask]

                if subcluster_data.shape[0] > 0:
                    N_sub = subcluster_data.shape[0]
                    # Pass K-means center for scatter computation
                    posterior = prior_cpu.compute_posterior(
                        subcluster_data,
                        cluster_mean=mus_sub_kmeans[sub_idx]
                    )
                    mu_sub, cov_sub = posterior.map_estimate()
                    new_mus_sub[2*cluster_idx1 + sub_idx] = mu_sub.to(device)
                    new_covs_sub[2*cluster_idx1 + sub_idx] = cov_sub.to(device)
                    new_pi_sub[2*cluster_idx1 + sub_idx] = (
                        submask.sum().float() / data.shape[0]
                    )
                else:
                    new_mus_sub[2*cluster_idx1 + sub_idx] = merged_mu
                    new_covs_sub[2*cluster_idx1 + sub_idx] = merged_cov
                    new_pi_sub[2*cluster_idx1 + sub_idx] = merged_pi / 2
        else:
            # Fallback: not enough data, use merged mean (no perturbation)
            new_mus_sub[2*cluster_idx1] = merged_mu
            new_mus_sub[2*cluster_idx1 + 1] = merged_mu
            new_covs_sub[2*cluster_idx1] = merged_cov
            new_covs_sub[2*cluster_idx1 + 1] = merged_cov
            new_pi_sub[2*cluster_idx1] = merged_pi / 2
            new_pi_sub[2*cluster_idx1 + 1] = merged_pi / 2

        # Between
        new_mus_sub[2*(cluster_idx1+1):2*cluster_idx2] = \
            self.mus_sub[2*(cluster_idx1+1):2*cluster_idx2]
        new_covs_sub[2*(cluster_idx1+1):2*cluster_idx2] = \
            self.covs_sub[2*(cluster_idx1+1):2*cluster_idx2]
        new_pi_sub[2*(cluster_idx1+1):2*cluster_idx2] = \
            self.pi_sub[2*(cluster_idx1+1):2*cluster_idx2]

        # After
        new_mus_sub[2*cluster_idx2:] = self.mus_sub[2*(cluster_idx2+1):]
        new_covs_sub[2*cluster_idx2:] = self.covs_sub[2*(cluster_idx2+1):]
        new_pi_sub[2*cluster_idx2:] = self.pi_sub[2*(cluster_idx2+1):]

        self.mus_sub = new_mus_sub
        self.covs_sub = new_covs_sub
        self.pi_sub = new_pi_sub

    @staticmethod
    def _mvn_log_prob(
        x: torch.Tensor,
        mu: torch.Tensor,
        cov: torch.Tensor
    ) -> torch.Tensor:
        """Compute multivariate normal log-probability.

        Args:
            x: (N, d) data points
            mu: (d,) mean
            cov: (d, d) covariance

        Returns:
            (N,) log-probabilities
        """
        d = mu.shape[0]
        device = x.device

        # Add small regularization to covariance for numerical stability
        cov_reg = cov + torch.eye(d, device=device) * 1e-6

        try:
            # Compute Cholesky decomposition
            L = torch.linalg.cholesky(cov_reg)

            # Compute log determinant
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))

            # Solve L @ L.T @ alpha = (x - mu).T
            diff = x - mu  # (N, d)
            alpha = torch.linalg.solve_triangular(L, diff.T, upper=False)  # (d, N)

            # Mahalanobis distance
            mahal = torch.sum(alpha ** 2, dim=0)  # (N,)

            # Log probability
            log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)

            return log_prob

        except RuntimeError:
            # Fallback: use diagonal approximation if Cholesky fails
            cov_diag = torch.diag(cov) + 1e-6
            log_det = torch.sum(torch.log(cov_diag))
            diff = x - mu
            mahal = torch.sum((diff ** 2) / cov_diag, dim=1)
            log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
            return log_prob
