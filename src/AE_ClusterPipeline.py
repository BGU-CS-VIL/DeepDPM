#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import numpy as np
import torch.nn as nn
from datetime import datetime

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import pytorch_lightning as pl

from src.clustering_models.clusternet import ClusterNet
from src.feature_extractors.feature_extractor import FeatureExtractor
from src.clustering_models.clusternet_modules.utils.training_utils import training_utils
from src.clustering_models.clusternet_modules.utils.plotting_utils import PlotUtils


'''
Based on the implementation from (https://github.com/xuyxu/Deep-Clustering-Network)
to "Towards k-means-friendly spaces: Simultaneous deep learning and clustering", Yang et al. ICML 2017:
http://proceedings.mlr.press/v70/yang17b/yang17b.pdf.

'''

class AE_ClusterPipeline(pl.LightningModule):
    def __init__(self, logger, args):
        super(AE_ClusterPipeline, self).__init__()
        self.args = args
        self.pretrain_logger = logger
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term
        self.lambda_ = args.lambda_  # coefficient of the reconstruction term
        self.n_clusters = self.args.n_clusters
        if self.args.seed:
            pl.utilities.seed.seed_everything(self.args.seed)

        # Validation check
        if not self.beta > 0:
            msg = "beta should be greater than 0 but got value = {}."
            raise ValueError(msg.format(self.beta))

        if not self.lambda_ > 0:
            msg = "lambda should be greater than 0 but got value = {}."
            raise ValueError(msg.format(self.lambda_))

        if len(self.args.hidden_dims) == 0:
            raise ValueError("No hidden layer specified.")

        self.feature_extractor = FeatureExtractor(args)
        self.args.latent_dim = self.feature_extractor.latent_dim
        self.criterion = nn.MSELoss(reduction="sum")

        self.clustering = ClusterNet(args, self)
        self.init_clusternet_num = 0  # number of times the clustering net was initialized
        self.plot_utils = PlotUtils(hparams=self.args)

    def configure_optimizers(self):
        """Configure the optimizers of the AE model
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.wd
        )
        return optimizer

    def _loss(self, X, latent_X, cluster_assignment):
        """Compute batch loss

        Args:
            X ([torch.tensor]): The current batch of data ([N, D])
            cluster_assignment ([torch.tensor]): (soft) cluster assignments for each data point

        Returns:
            Tuple: the loss (reconstruction + distance loss) and both losses seperately for logging
        """
        batch_size = X.size()[0]
        rec_X = self.feature_extractor.decode(latent_X)
        # latent_X = self.autoencoder(X, latent=True)

        # Reconstruction error
        X = self.feature_extractor.extract_features(X) if self.feature_extractor.feature_extractor else X
        rec_loss = self.lambda_ * self.criterion(X, rec_X)

        # Regularization term on clustering
        if self.args.regularization == "dist_loss":
            dist_loss = torch.tensor(0.0).to(self.device)
            clusters = torch.FloatTensor(self.clustering.clusters).to(self.device)
            for i in range(batch_size):
                diff_vec = latent_X[i] - clusters[cluster_assignment.argmax(-1)[i]]
                sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
                dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)
            reg_loss = dist_loss

        elif self.args.regularization == "cluster_loss":
            # const clustering variables (w.r.t this training stage)
            mus, covs, pi, K = self.clustering.get_model_params()

            reg_loss = self.clustering.model.cluster_model.training_utils.cluster_loss_function(
                latent_X.detach(),
                cluster_assignment,
                model_mus=torch.from_numpy(mus),
                K=K,
                codes_dim=self.args.latent_dim,
                model_covs=torch.from_numpy(covs) if self.args.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
                pi=torch.from_numpy(pi)
            ) * batch_size

        return (
            rec_loss + reg_loss,
            rec_loss.detach(),
            reg_loss.detach(),
        )

    def _init_clusters(self, verbose=True, centers=None):
        if verbose:
            print(f"========== Alternation {self.init_clusternet_num}: Running DeepDPM clustering ==========\n")
        if self.args.clustering == "cluster_net":
            self.clustering.init_cluster(self.train_dataloader(), self.val_dataloader(), logger=self.logger, centers=centers, init_num=self.init_clusternet_num)
            self.n_clusters = self.clustering.n_clusters
            if self.args.save_checkpoints:
                # Checkpoint
                print('Checkpoint ...')
                import os
                save_dict = {}
                clustering_module = self.clustering.model.cluster_model
                if len(clustering_module.optimizers_dict_idx) > 1:
                    for key, value in clustering_module.optimizers_dict_idx.items():
                        save_dict[key] = clustering_module.optimizers()[value].state_dict()
                else:
                    save_dict["clusternet_opt"] = clustering_module.optimizers().state_dict()
                save_dict['model'] = clustering_module.state_dict(),
                save_dict['K'] = clustering_module.K,
                save_dict['epoch'] = clustering_module.current_epoch
                save_dict['alt_num'] = self.init_clusternet_num
                torch.save(save_dict, f'./saved_models/{self.args.dataset}/{self.args.exp_name}/alt_{self.init_clusternet_num}_checkpoint.pth.tar')
            self.init_clusternet_num += 1
            self.log("alt_num", self.init_clusternet_num)

        else:
            batch_X, batch_Y = [], []
            for data, labels in self.train_dataloader():
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                latent_X = self.feature_extractor(data, latent=True)
                batch_X.append(latent_X.detach().cpu().numpy())
                batch_Y.append(labels)
            batch_X = np.vstack(batch_X)
            batch_Y = torch.cat(batch_Y)
            self.clustering.init_cluster(batch_X)
            y_pred = self.clustering.update_assign(torch.from_numpy(batch_X))
            init_nmi = normalized_mutual_info_score(batch_Y.numpy(), y_pred.argmax(-1))
            self.log("train/k_means_init_nmi", init_nmi)

        if verbose:
            print("========== End initializing clusters ==========\n")
        


    def _comp_clusters(self, verbose=True):
        used_centers_for_initialization = self.clustering.clusters if self.args.init_cluster_net_using_centers else None
        if self.args.reinit_net_at_alternation or self.args.clustering != "cluster_net":
            self._init_clusters(verbose, centers=used_centers_for_initialization)
        else:
            self.clustering.fit_cluster(self.train_dataloader(), self.val_dataloader(), logger=self.logger, centers=used_centers_for_initialization)
            self.n_clusters = self.clustering.n_clusters

    def _pre_step(self, x, y=None):
        if len(self.sampled_codes) < 10000:
            with torch.no_grad():
                latent_X = self.feature_extractor(x, latent=True)
                self.sampled_codes = torch.cat([self.sampled_codes, latent_X.detach().cpu()])
                self.sampled_gt = torch.cat([self.sampled_gt, y.cpu()])
        rec_X = self.feature_extractor(x)
        x = self.feature_extractor.extract_features(x) if self.feature_extractor.feature_extractor else x
        loss = self.criterion(x, rec_X)
        return loss

    def _step(self, x, y):
        """Implementing one optimization step.
        1. gets the latent features using a forward pass on the AE
        2. gets the cluster assignments for the latent features (the closest cluster id to each sample is chosen)
        3. updates the clusters centers in an online fashion (eta = 1 / (N_k + x_i) , (1-eta)*mu + eta*x_i)
        4. Compute and return loss

        Args:
            x (torch.tensor): batch [N, D]

        Returns:
            loss and losses for logging
        """
        # Get the latent features
        # latent_X = self(x, latent=True) # constant w.r.t to net weights
        # Update the assignments
        latent_X, cluster_assign = self(x)
        # save for plotting
        if len(self.sampled_codes) < 10000:
            self.sampled_codes = torch.cat([self.sampled_codes, latent_X.detach().cpu()])
            self.sampled_gt = torch.cat([self.sampled_gt, y.cpu()])

        if self.args.update_clusters_params != "False":
            # [Step-2] Update clusters in batch Clustering
            self._update_clusters(latent_X, cluster_assign)

        # Update the network parameters
        loss, rec_loss, dist_loss = self._loss(x, latent_X, cluster_assign)
        return loss, rec_loss, dist_loss

    def _update_clusters(self, latent_X, cluster_assign):
        if self.args.update_clusters_params == "only_centers":
            elem_count = cluster_assign.sum(axis=0)
            for k in range(self.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.clustering.update_cluster_center(latent_X.detach().cpu().numpy(), k, cluster_assign.detach().cpu().numpy())

    def forward(self, x, latent=False):
        # Get the latent features
        latent_X = self.feature_extractor(x, latent=True)
        if latent:
            # used by the clusternet. net is frozen so no grad accumulated
            return latent_X.detach().cpu().numpy()
        # [Step-1] Update/Get the assignment results - returns a vector sized N_samples X N_classes (could be one-hot)
        if len(latent_X.size()) > 2:
            latent_X = latent_X.view(latent_X.size(0), -1)
        if self.args.cluster_assignments != "pseudo_label":
            return latent_X, self.clustering.update_assign(latent_X, self.args.cluster_assignments)
        else:
            return 0

    def on_train_epoch_start(self):

        # for plotting
        self.sampled_codes = torch.empty(0)
        self.sampled_gt = torch.empty(0)
        if self.current_epoch == 0:
            self.log('data_stats/train_n_samples', len(self.train_dataloader().dataset))
            self.log('data_stats/val_n_samples', len(self.val_dataloader().dataset))
            if self.args.pretrain:
                assert self.args.pretrain_epochs > 0
                print("========== Start pretraining ==========")
                self.pretrain = True
            else:
                self.pretrain = False
                # using pretrained weights only initialize clusters
                assert self.args.pretrain_path is not None
                self._init_clusters()

        elif self.args.pretrain and self.pretrain and self.current_epoch in (5, 10, 15, 20, 25, 30, 35, 40, 50, 100, 500):
            print("Saving weights...")
            torch.save(self.state_dict(), f"./saved_models/{self.args.dataset}_{self.current_epoch}_{datetime.now()}")

        elif self.current_epoch == self.args.pretrain_epochs and self.args.pretrain and self.pretrain:
            print("========== End pretraining ==========")
            self.pretrain = False
            print("Saving weights...")
            torch.save(self.state_dict(), f"./saved_models/{self.args.dataset}_{self.current_epoch}_{datetime.now()}")
            print("Great success")
            self._init_clusters()
        if self.args.alternate:
            if self.current_epoch >= self.args.epoch and self.current_epoch % self.args.retrain_cluster_net_every == 0 and not self.pretrain:
                self._comp_clusters()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        if self.pretrain:
            stage = "val_pretrain"
            loss = self._pre_step(x, y)
            rec_loss = loss
            dist_loss = torch.tensor([0.])
        else:
            stage = "val"
            loss, rec_loss, dist_loss = self._step(x, y)
        self.log(f"{stage}/loss", loss)
        self.log(f"{stage}/reconstruction_loss", rec_loss)
        self.log(f"{stage}/dist_loss", dist_loss)

        _, assign = self(x)
        y_pred = assign.argmax(-1).cpu().numpy()
        return {"loss": loss, "y_gt": y, "y_pred": y_pred}

    def training_step(self, batch, batch_idx,):
        x, y = batch
        x = x.view(x.size(0), -1)

        if self.pretrain:
            stage = "pretrain"
            if self.args.pretrain_noise_factor > 0:
                x = x + self.args.pretrain_noise_factor * torch.randn(*x.shape).to(device=self.device)
                # Clip the images to be between 0 and 1
                x = np.clip(x.cpu(), 0., 1.).to(device=self.device)
            loss = self._pre_step(x, y)
            rec_loss = loss
            dist_loss = torch.tensor([0.])

        else:
            stage = "train"
            loss, rec_loss, dist_loss = self._step(x, y)
        self.log(f"{stage}/loss", loss)
        self.log(f"{stage}/reconstruction_loss", rec_loss)
        self.log(f"{stage}/dist_loss", dist_loss)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        losses, y_gt, y_pred = [], [], []
        for out_dict in validation_step_outputs:
            losses.append(out_dict['loss'])
            y_gt += list(out_dict['y_gt'].cpu().numpy())
            y_pred += list(out_dict['y_pred'])

        avg_loss = torch.tensor(losses).mean()
        y_pred = np.vstack(y_pred).reshape(-1)

        NMI = normalized_mutual_info_score(y_gt, y_pred)
        ARI = adjusted_rand_score(y_gt, y_pred)
        ACC_top5, ACC = training_utils.cluster_acc(torch.tensor(y_gt), torch.from_numpy(y_pred))
        self.log("val/avg_loss", avg_loss)
        self.log("val/NMI", NMI)
        self.log("val/ARI", ARI)
        self.log("val/ACC", ACC)

        if not self.pretrain and self.args.log_emb != "never" and self.current_epoch in (0, 5, 10, 50, 100, 200, 400, 499, 500, 600, 700, 800, 900, 100):
            self.plot_utils.visualize_embeddings(
                self.args,
                self.logger,
                self.args.latent_dim,
                vae_means=self.sampled_codes,
                vae_labels=self.sampled_gt,
                val_resp=None,
                current_epoch=self.current_epoch,
                y_hat=None,
                centers=None,
                training_stage='train_AE',
                UMAP=False
            )

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--lambda_",
            type=float,
            default=0.005,
            help="coefficient of the reconstruction loss",
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=1,
            help="coefficient of the regularization term on " "clustering",
        )
        parser.add_argument(
            "--hidden-dims", type=int, nargs="+", default=[500, 500, 2000], help="hidden AE dims"
        )
        parser.add_argument(
            "--latent_dim", type=int, default=10, help="latent space dimension"
        )
        parser.add_argument(
            "--n-clusters",
            type=int,
            default=10,
            help="number of clusters in the latent space",
        )
        parser.add_argument(
            "--pretrain_noise_factor", type=float, default=0, help="the noise factor to be used in pretraining" 
        )
        parser.add_argument(
            "--clustering",
            type=str,
            default="cluster_net",
            help="choose a clustering method",
        )
        parser.add_argument(
            "--alternate",
            action="store_true"
        )
        parser.add_argument(
            "--retrain_cluster_net_every",
            type=int,
            default=100,
        )
        parser.add_argument(
            "--init_cluster_net_using_centers",
            action="store_true"
        )
        parser.add_argument(
            "--reinit_net_at_alternation",
            action="store_true"
        )
        parser.add_argument(
            "--regularization",
            type=str,
            choices=["dist_loss", "cluster_loss"],
            help="which cluster regularization to use on the AE",
            default="dist_loss"
        )
        parser.add_argument(
            "--cluster_assignments",
            type=str,
            help="how to get the cluster assignment while training the AE, min_dist (hard assignment), forward_pass (soft assignment), pseudo_label (hard/soft assignment, TBD)",
            choices=["min_dist", "forward_pass", "pseudo_label"],
            default="min_dist"
        )
        parser.add_argument(
            "--update_clusters_params",
            type=str,
            choices=["False", "only_centers", "all_params", "all_params_w_prior"],
            default="False",
            help="whether and how to update the clusters params (e.g., center) during the AE training"
        )
        return parser
