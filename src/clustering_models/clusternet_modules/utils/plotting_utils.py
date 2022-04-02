#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

from sklearn.manifold import TSNE
import umap
from matplotlib import pyplot as plt

import numpy as np
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib as mpl

import pandas as pd

class PlotUtils:
    def __init__(self, hparams, logger=None, samples=None):
        self.mus_ind_merge = None
        self.mus_ind_split = None
        self.hparams = hparams
        self.logger = logger
        self.cmap = mpl.colors.ListedColormap(np.random.rand(100, 3))
        self.colors = None
        self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"

    def visualize_embeddings(self, hparams, logger, codes_dim, vae_means, vae_labels=None, val_resp=None, current_epoch=None, UMAP=True, EM_labels=None, y_hat=None, stage="cluster_net_train", centers=None, training_stage='val'):
        method = "UMAP" if UMAP else "TSNE"
        if codes_dim > 2:
            if training_stage != "val_thesis" or (training_stage == "val_thesis" and not hasattr(self, 'val_embb')):
                print("Transforming using UMAP/TSNE...")
                if UMAP:
                    umap_obj = umap.UMAP(
                        n_neighbors=30,
                        min_dist=0.0,
                        n_components=2,
                        random_state=42,
                    ).fit(vae_means.detach().cpu())
                    E = umap_obj.embedding_
                    if centers is not None:
                        centers = umap_obj.transform(centers.cpu())
                else:
                    E = TSNE(n_components=2).fit_transform(vae_means.detach().cpu())
        else:
            E = vae_means.detach().cpu()
        if val_resp is not None:
            if training_stage != "val_thesis":
                fig = plt.figure(figsize=(16, 10))
                plt.scatter(E[:, 0], E[:, 1], c=val_resp.argmax(-1), cmap="tab10")
                if centers is not None:
                    plt.scatter(centers[:, 0], centers[:, 1], c=np.arange(len(centers)), marker='*', edgecolor='k')
                plt.title(f"{method} embeddings, epoch {current_epoch}")
                from pytorch_lightning.loggers.base import DummyLogger
                if not isinstance(self.logger, DummyLogger):
                    logger.log_image(f"{stage}/{training_stage}/{method} embeddings using net labels", fig)
                plt.close(fig)

            else:
                if hasattr(self, 'val_embb'):
                    E = self.val_embb
                else:
                    self.val_embb = E

                fig = plt.figure(figsize=(16, 10))
                plt.scatter(E[:, 0], E[:, 1], c=val_resp.argmax(-1), cmap="tab10")
                ax = plt.gca()
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                from pytorch_lightning.loggers.base import DummyLogger
                if not isinstance(self.logger, DummyLogger):
                    logger.log_image(f"{stage}/{training_stage}/{method} embeddings using net labels new", fig)
                plt.close(fig)

        if y_hat is not None:
            fig = plt.figure(figsize=(16, 10))
            plt.scatter(E[:, 0], E[:, 1], c=y_hat, cmap="tab10")
            plt.title(f"{method} embeddings, epoch {current_epoch}")

            from pytorch_lightning.loggers.base import DummyLogger
            if not isinstance(self.logger, DummyLogger):
                logger.log_image(f"{stage}/{training_stage}/{method} embeddings using net pseudo-labels", fig)
            plt.close(fig)

        fig = plt.figure(figsize=(16, 10))
        labels = vae_labels if EM_labels is None else EM_labels
        plt.scatter(E[:, 0], E[:, 1], c=labels, cmap="tab10")
        if centers is not None and len(np.unique(labels)) == len(centers):
            plt.scatter(centers[:, 0], centers[:, 1], c=sorted(np.unique(labels)), marker='*', edgecolor='k')
        if training_stage == "val_thesis":
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        else:
            plt.title(f"{method} embeddings, epoch {current_epoch}")

        from pytorch_lightning.loggers.base import DummyLogger
        if not isinstance(logger, DummyLogger):
            if EM_labels is None:
                logger.log_image(f"{stage}/{training_stage}/{method} embeddings using true labels", fig)
            else:
                logger.log_image(f"{stage}/{training_stage}/{method} embeddings using EM labels", fig)
        plt.close(fig)

    def visualize_embeddings_old(data, labels, use_pca_first=False):
        x = data.detach().cpu()
        if use_pca_first:
            print("Performing PCA...")
            pca_50 = PCA(n_components=50)
            data = pca_50.fit_transform(x)

        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=1, n_iter=1000, metric="cosine")
        data_2d = tsne.fit_transform(x)
        fig = plt.figure(figsize=(6, 5))
        num_classes = len(np.unique(labels))
        palette = np.array(sns.color_palette("hls", num_classes))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], lw=0, s=40, c=palette[labels])
        plt.legend()
        plt.show()
        return fig

    def embed_to_2d(self, data):
        if data.shape[1] > 50:
            print("Performing PCA...")
            pca_50 = PCA(n_components=50)
            data = pca_50.fit_transform(data)
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=1, n_iter=1000, metric="cosine")
        data_2d = tsne.fit_transform(data)
        return data_2d

    def sklearn_make_ellipses(self, center, cov, ax, color, **kwargs):
        covariance = cov  # np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariance)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(np.abs(v))
        ell = mpl.patches.Ellipse(center, v[0], v[1], 180 + angle, color=color, **kwargs)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        return ell

    def draw_ellipse(self, x, y, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse((x, y), nsig * width, nsig * height, angle, **kwargs))

    def plot_clusters_colored_by_label(self, samples, y_gt, n_epoch, K):
        return self.plot_clusters_by_color(samples, y_gt, n_epoch, K, labels_type="gt")

    def plot_clusters_colored_by_net(self, samples, y_net, n_epoch, K):
        return self.plot_clusters_by_color(samples, y_net, n_epoch, K, labels_type="net")

    def plot_clusters_by_color(self, samples, labels, n_epoch, K, labels_type):
        fig = plt.figure(figsize=(16, 10))
        df = pd.DataFrame(columns=['x_pca', 'y_pca', 'label'])
        samples_pca = self.pca.transform(samples)
        df['x_pca'] = samples_pca[:, 0]
        df['y_pca'] = samples_pca[:, 1]
        df['label'] = labels
        df['label'] = df['label'].astype(str)
        sns.scatterplot(
            x="x_pca", y="y_pca",
            hue="label",
            # color=self.colors['']
            palette=sns.color_palette("hls", K),
            data=df,
            legend="full",
            alpha=0.3,
        )
        plt.title(f"Epoch {n_epoch}: pca-ed data colored by {labels_type} labels")
        return fig

    def plot_decision_regions(
        self,
        X,
        cluster_net,
        ax,
        y_gt
    ):
        X_min, X_max = X.min(axis=0).values, X.max(axis=0).values 
        arrays_for_meshgrid = [np.arange(X_min[d] - 0.1, X_max[d] + 0.1, 0.1) for d in range(X.shape[1])]
        xx, yy = np.meshgrid(*arrays_for_meshgrid)

        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        # horizontal stack vectors to create x1,x2 input for the model
        grid = np.hstack((r1,r2))
        yhat = cluster_net(torch.from_numpy(grid).float().to(self.device))
        yhat_maxed = yhat.max(axis=1).values.cpu()

        cont = ax.contourf(xx, yy, yhat_maxed.reshape(xx.shape), alpha=0.5, cmap="jet")

        ax.scatter(
            X[:, 0],
            X[:, 1],
            cmap="tab20",
            c=y_gt,
        )
        ax.set_title("Decision boundary \n Clusters are colored by GT labels")
        return cont

    def plot_clusters(
        self,
        ax_clusters,
        samples,
        labels,
        centers,
        covs,
        sub_center,
        sub_covs,
        mu_gt = None,
        n_epoch=None,
        alone=False,
    ):
        # expects to get samples, a 2-dim vector and labels are the true labels
        # centers are the centers that were found by the classifier

        if self.colors is None:
            # first time
            self.colors = torch.rand(self.hparams.init_k, 3)

        if alone:
            mu_gt = np.array([x.numpy() for x in mu_gt])
            if samples.shape[1] > 2:
                # perform PCA
                print("Performing PCA...")
                samples = self.pca.transform(samples)
                centers = self.pca.transform(centers)
                mu_gt = self.pca.transform(mu_gt)
                covs_pca, sub_covs_pca = [], []
                for cov in covs:
                    cov_diag = torch.tensor(cov).diag()
                    covs_pca.append(torch.eye(2) * self.pca.transform(cov_diag.reshape(1, -1)))
                covs = covs_pca
                if sub_covs is not None:
                    for sub_cov in sub_covs:
                        cov_diag = torch.tensor(sub_cov).diag()
                        sub_covs_pca.append(torch.eye(2) * self.pca.transform(cov_diag.reshape(1, -1)))
                    sub_covs = sub_covs_pca
            fig_clusters, ax_clusters = plt.subplots()

        # plot points colored by the given labels
        ax_clusters.scatter(
            samples[:, 0], samples[:, 1], c=self.colors[labels, :], s=40, alpha=0.5, zorder=1
        )

        # plot the gt centers and the net's centers
        if mu_gt:
            ax_clusters.plot(
                mu_gt[:, 0], mu_gt[:, 1], "g*", label="Real centers", markersize=15.0, zorder=2
            )
        ax_clusters.plot(
            centers[:, 0],
            centers[:, 1],
            "ko",
            label="net centers",
            markersize=12.0,
            alpha=0.6,
            zorder=3
        )

        # plot covs
        for i, center in enumerate(centers):
            ell = self.sklearn_make_ellipses(center=center, cov=covs[i], ax=ax_clusters, color=self.colors[i].numpy())
            ax_clusters.add_artist(ell)

        # plot net's subclusters
        if sub_center is not None:
            ax_clusters.scatter(
                sub_center[:, 0],
                sub_center[:, 1],
                marker='*',
                c=self.colors[np.arange(len(centers)).repeat(2)],
                edgecolors='k',
                label="net subcenters",
                s=100.0,
                alpha=1,
                zorder=4
            )
            for j, sub in enumerate(sub_center):
                ax_clusters.text(sub[0]+0.03, sub[1], str(j % 2), c='k', fontsize=12)  # self.colors[j // 2].numpy(), fontsize=12)

            for i in range(len(sub_center)):
                ell = self.sklearn_make_ellipses(center=sub_center[i], cov=sub_covs[i], ax=ax_clusters, color=self.colors[i//2].numpy(), ls="--", fill=False)
                ax_clusters.add_artist(ell)

        ax_clusters.set_title("Net centers and covariances \n Clusters are colored by net's assignments")
        # ax_clusters.legend()
        if alone:
            ax_clusters.set_title(
                f"Epoch {n_epoch}: Clusters colored by net's assignments"
            )
            return fig_clusters

    def plot_cluster_and_decision_boundaries(
        self,
        samples,
        labels,
        gt_labels,
        net_centers,
        net_covs,
        n_epoch,
        cluster_net,
        gt_centers=None,
        net_sub_centers=None,
        net_sub_covs=None,
    ):
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 8))
        # fig.tight_layout()
        if gt_centers:
            gt_centers = np.array([x.numpy() for x in gt_centers])

        # set aspect ratio
        _min, _max = samples.min(axis=0).values, samples.max(axis=0).values 
        (ax_clusters, ax_boundaries) = axes

        self.plot_clusters(
            ax_clusters=ax_clusters,
            samples=samples,
            labels=labels,
            centers=net_centers,
            covs=net_covs,
            mu_gt=gt_centers,
            sub_center=net_sub_centers,
            sub_covs=net_sub_covs,
        )

        cont_for_color_bar = self.plot_decision_regions(
                X=samples, cluster_net=cluster_net, ax=ax_boundaries, y_gt=gt_labels)

        for ax in axes:
            ax.set_xlim([_min[0], _max[0]])
            ax.set_ylim([_min[1], _max[1]])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            
        cbar = fig.colorbar(cont_for_color_bar, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("Max network response", rotation=270, labelpad=10, y=0.45)

        # fig.suptitle(f"Epoch: {n_epoch}", fontsize=14, weight="bold")

        import os
        if not os.path.exists("./imgs/"):
            os.makedirs("./imgs/")
            os.makedirs("./imgs/clusters/")
            os.makedirs("./imgs/decision_boundary/")
        fig.savefig(f"./imgs/{n_epoch}.png")

        # save just the clusters fig
        extent_clus = ax_clusters.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        extent_bound = ax_boundaries.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        if cluster_net.training:
            fig.savefig(
                f"./imgs/clusters/{n_epoch}.png",
                bbox_inches=extent_clus,
            )
            fig.savefig(
                f"./imgs/decision_boundary/{n_epoch}.png",
                bbox_inches=extent_bound,
            )

        plt.close()
        return

    def plot_weights_histograms(self, K, pi, start_sub_clustering, current_epoch, pi_sub, for_thesis=False):
        fig = plt.figure(figsize=(10, 3))
        ind = np.arange(K)
        plt.bar(ind, pi, label="clusters' weights", align="center", alpha=0.3)
        if start_sub_clustering <= current_epoch and pi_sub is not None:
            pi_sub_1 = pi_sub[0::2]
            pi_sub_2 = pi_sub[1::2]
            plt.bar(ind, pi_sub_1, align="center", label="sub cluster 1")
            plt.bar(
                ind, pi_sub_2, align="center", bottom=pi_sub_1, label="sub cluster 2"
            )

        plt.xlabel("Clusters inds")
        plt.ylabel("Normalized weights")
        plt.title(f"Epoch {current_epoch}: Clusters weights")
        if for_thesis:
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        else:
            plt.legend()
        return fig

    def plot_cov_eig_values(self, covs_to_plot, epoch):
        for i in range(len(covs_to_plot)):
            e, _ = torch.torch.linalg.eig(covs_to_plot[i])
            e = torch.real(e)
            fig = plt.figure(figsize=(16, 10))
            plt.plot(range(len(e)), sorted(e, reverse=True))
            plt.title(f"the eigenvalues of cov {i} to be split, epoch {epoch}")
            plt.xlabel("Eigenvalues inds")
            plt.ylabel("Eigenvalues")
            self.logger.log_image(f"cluster_net_train/train/epoch {epoch}/eigenvalues_cov_{i}", fig)
            plt.close(fig)

    def update_colors(self, split, split_inds, merge_inds):
        if split:
            self.update_colors_split(split_inds)
        else:
            self.update_colors_merge(merge_inds)


    def update_colors_split(self, mus_ind_split):
        mask = torch.zeros(len(self.colors), dtype=bool)
        mask[mus_ind_split.flatten()] = 1
        colors_not_split = self.colors[torch.logical_not(mask)]
        colors_split = self.colors[mask].repeat(1, 2).view(-1, 3)
        colors_split[1::2] = torch.rand(len(colors_split[1::2]), 3)
        self.colors = torch.cat([colors_not_split, colors_split])

    def update_colors_merge(self, mus_ind_merge):
        mask = torch.zeros(len(self.colors))
        mask[mus_ind_merge.flatten()] = 1
        colors_not_merged = self.colors[torch.logical_not(mask)]
        # take all the non merges clusters' colors and the color of the first index of a merge pair
        self.colors = torch.cat(
            [colors_not_merged, self.colors[mus_ind_merge[:, 0]]]
        )


