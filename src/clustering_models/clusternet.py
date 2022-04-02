#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import numpy as np
from joblib import Parallel, delayed
import torch

from src.clustering_models.clusternet_modules.clusternet_trainer import (
    ClusterNetTrainer,
)


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class ClusterNet(object):
    def __init__(self, args, feature_extractor):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs
        self.feature_extractor = feature_extractor
        self.device = "cuda" if torch.cuda.is_available() and args.gpus is not None else "cpu"

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters)
        )
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, train_loader, val_loader, logger, indices=None, centers=None, init_num=0):
        """ Generate initial clusters using the clusternet
            init num is the number of time the clusternet was initialized (from the AE_ClusterPipeline module)
        """
        self.feature_extractor.freeze()
        self.model = ClusterNetTrainer(
            args=self.args,
            init_k=self.n_clusters,
            latent_dim=self.latent_dim,
            feature_extractor=self.feature_extractor,
            centers=centers,
            init_num=init_num
        )
        self.fit_cluster(train_loader, val_loader, logger, centers)
        self.model.cluster_model.freeze()
        self.feature_extractor.unfreeze()
        self.feature_extractor.to(device=self.device)

    def fit_cluster(self, train_loader, val_loader, logger, centers=None):
        self.feature_extractor.freeze()
        self.model.cluster_model.unfreeze()
        self.model.fit(train_loader, val_loader, logger, self.args.train_cluster_net, centers=centers)
        self.model.cluster_model.freeze()
        self.clusters = self.model.get_clusters_centers()  # copy clusters
        self._set_K(self.model.get_current_K())
        self.feature_extractor.unfreeze()
        self.feature_extractor.to(device=self.device)

    def freeze(self):
        self.model.cluster_model.freeze()
        self.feature_extractor.unfreeze()

    def unfreeze(self):
        self.model.cluster_model.unfreeze()
        self.model.cluster_model.to(device=self.device)

    def update_cluster_center(self, X, cluster_idx, assignments=None):
        """ Update clusters centers on a batch of data

        Args:
            X (torch.tensor): All the data points that were assigned to this cluster
            cluster_idx (int): The cluster index
            assignments: The probability of each cluster to be assigned to this cluster (would be a vector of ones for hard assignment)
        """
        n_samples = X.shape[0]
        for i in range(n_samples):
            if assignments[i, cluster_idx].item() > 0:
                self.count[cluster_idx] += assignments[i, cluster_idx].item()
                eta = 1.0 / self.count[cluster_idx]
                updated_cluster = (1 - eta) * self.clusters[cluster_idx] + eta * X[i] * assignments[i, cluster_idx].item()
                # updated_cluster = (1 - eta) * self.clusters[cluster_idx] + eta * X[i]
                self.clusters[cluster_idx] = updated_cluster

    def update_cluster_covs(self, X, cluster_idx, assignments):
        return None

    def update_cluster_pis(self, X, cluster_idx, assignments):
        return None

    def update_assign(self, X, how_to_assign="min_dist"):
        """ Assign samples in `X` to clusters """
        if how_to_assign == "min_dist":
            return self._update_assign_min_dist(X.detach().cpu().numpy())
        elif how_to_assign == "forward_pass":
            return self.get_model_resp(X)

    def _update_assign_min_dist(self, X):
        dis_mat = self._compute_dist(X)
        hard_assign = np.argmin(dis_mat, axis=1)
        return self._to_one_hot(torch.tensor(hard_assign))

    def _to_one_hot(self, hard_assignments):
        """
        Takes LongTensor with index values of shape (*) and
        returns a tensor of shape (*, num_classes) that have zeros everywhere
        except where the index of last dimension matches the corresponding value
        of the input tensor, in which case it will be 1.
        """
        return torch.nn.functional.one_hot(hard_assignments, num_classes=self.n_clusters)

    def _set_K(self, new_K):
        self.n_clusters = new_K
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate, pseudo-counts

    def get_model_params(self):
        mu, covs, pi, K = self.model.get_clusters_centers(), self.model.get_clusters_covs(), self.model.get_clusters_pis(), self.n_clusters
        return mu, covs, pi, K

    def get_model_resp(self, codes):
        self.model.cluster_model.to(device=self.device)
        if self.args.regularization == "cluster_loss":
            # cluster assignment should have grad
            return self.model.cluster_model(codes)
        else:
            # cluster assignment shouldn't have grad
            with torch.no_grad():
                return self.model.cluster_model(codes)
