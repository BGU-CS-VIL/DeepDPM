#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import argparse
from argparse import ArgumentParser
import os
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
import pytorch_lightning as pl
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np

from src.embbeded_datasets import embbededDataset
from src.datasets import GMM_dataset
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel


def parse_minimal_args(parser):
    # Dataset parameters
    parser.add_argument("--dir", default="./pretrained_embeddings/umap_embedded_datasets/", help="dataset directory")
    parser.add_argument("--dataset", default="MNIST_N2D")
    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="number of jobs to run in parallel"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device for computation (default: cpu)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run training without Neptune Logger"
    )
    parser.add_argument(
        "--tag", type=str, default="MNIST_UMAPED",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500,
    )
    parser.add_argument(
        "--latent_dim", type=int, default=10, help="the input data dim for DeepDPM"
    )
    parser.add_argument(
        "--limit_train_batches", type=float, default=1., help="used for debugging"
    )
    parser.add_argument(
        "--limit_val_batches", type=float, default=1., help="used for debugging" 
    )
    parser.add_argument(
        "--save_checkpoints", type=bool, default=False
    )
    parser.add_argument(
        "--exp_name", type=str, default="default_exp"
    )
    return parser

def run_on_embeddings_hyperparams(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--init_k", default=1, type=int, help="number of initial clusters"
    )
    parser.add_argument(
        "--clusternet_hidden",
        type=int,
        default=50,
        help="The dimensions of the hidden dim of the clusternet. Defaults to 50.",
    )
    parser.add_argument(
        "--clusternet_hidden_layer_list",
        type=int,
        nargs="+",
        default=[50],
        help="The hidden layers in the clusternet. Defaults to [50, 50].",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="normalize",
        choices=["normalize", "min_max", "standard", "standard_normalize", "None", None],
        help="Use normalization for embedded data",
    )
    parser.add_argument(
        "--cluster_loss_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--init_cluster_net_weights",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--when_to_compute_mu",
        type=str,
        choices=["once", "every_epoch", "every_5_epochs"],
        default="every_epoch",
    )
    parser.add_argument(
        "--how_to_compute_mu",
        type=str,
        choices=["kmeans", "soft_assign"],
        default="soft_assign",
    )
    parser.add_argument(
        "--how_to_init_mu",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans",
    )
    parser.add_argument(
        "--how_to_init_mu_sub",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans_1d",
    )
    parser.add_argument(
        "--log_emb_every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--log_emb",
        type=str,
        default="never",
        choices=["every_n_epochs", "only_sampled", "never"]
    )
    parser.add_argument(
        "--train_cluster_net",
        type=int,
        default=300,
        help="Number of epochs to pretrain the cluster net",
    )
    parser.add_argument(
        "--cluster_lr",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--subcluster_lr",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="StepLR", choices=["StepLR", "None", "ReduceOnP"]
    )
    parser.add_argument(
        "--start_sub_clustering",
        type=int,
        default=45,
    )
    parser.add_argument(
        "--subcluster_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--start_splitting",
        type=int,
        default=55,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--subcluster_softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--split_prob",
        type=float,
        default=None,
        help="Split with this probability even if split rule is not met.  If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--merge_prob",
        type=float,
        default=None,
        help="merge with this probability even if merge rule is not met. If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--init_new_weights",
        type=str,
        default="same",
        choices=["same", "random", "subclusters"],
        help="How to create new weights after split. Same duplicates the old cluster's weights to the two new ones, random generate random weights and subclusters copies the weights from the subclustering net",
    )
    parser.add_argument(
        "--start_merging",
        type=int,
        default=55,
        help="The epoch in which to start consider merge proposals",
    )
    parser.add_argument(
        "--merge_init_weights_sub",
        type=str,
        default="highest_ll",
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_init_weights_sub",
        type=str,
        default="random",
        choices=["same_w_noise", "same", "random"],
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_every_n_epochs",
        type=int,
        default=10,
        help="Example: if set to 10, split proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--split_merge_every_n_epochs",
        type=int,
        default=30,
        help="Example: if set to 10, split proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--merge_every_n_epochs",
        type=int,
        default=10,
        help="Example: if set to 10, merge proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--raise_merge_proposals",
        type=str,
        default="brute_force_NN",
        help="how to raise merge proposals",
    )
    parser.add_argument(
        "--cov_const",
        type=float,
        default=0.005,
        help="gmms covs (in the Hastings ratio) will be torch.eye * cov_const",
    )
    parser.add_argument(
        "--freeze_mus_submus_after_splitmerge",
        type=int,
        default=5,
        help="Numbers of epochs to freeze the mus and sub mus following a split or a merge step",
    )
    parser.add_argument(
        "--freeze_mus_after_init",
        type=int,
        default=5,
        help="Numbers of epochs to freeze the mus and sub mus following a new initialization",
    )
    parser.add_argument(
        "--use_priors",
        type=int,
        default=1,
        help="Whether to use priors when computing model's parameters",
    )
    parser.add_argument("--prior", type=str, default="NIW", choices=["NIW", "NIG"])
    parser.add_argument(
        "--pi_prior", type=str, default="uniform", choices=["uniform", None]
    )
    parser.add_argument(
        "--prior_dir_counts",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prior_kappa",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--prior_nu",
        type=float,
        default=12.0,
        help="Need to be at least codes_dim + 1",
    )
    parser.add_argument(
        "--prior_mu_0",
        type=str,
        default="data_mean",
    )
    parser.add_argument(
        "--prior_sigma_choice",
        type=str,
        default="isotropic",
        choices=["iso_005", "iso_001", "iso_0001", "data_std"],
    )
    parser.add_argument(
        "--prior_sigma_scale",
        type=float,
        default=".005",
    )
    parser.add_argument(
        "--prior_sigma_scale_step",
        type=float,
        default=1.,
        help="add to change sigma scale between alternations"
    )
    parser.add_argument(
        "--compute_params_every",
        type=int,
        help="How frequently to compute the clustering params (mus, sub, pis)",
        default=1,
    )
    parser.add_argument(
        "--start_computing_params",
        type=int,
        help="When to start to compute the clustering params (mus, sub, pis)",
        default=25,
    )
    parser.add_argument(
        "--cluster_loss",
        type=str,
        help="What kind og loss to use",
        default="KL_GMM_2",
        choices=["diag_NIG", "isotropic", "KL_GMM_2"],
    )
    parser.add_argument(
        "--subcluster_loss",
        type=str,
        help="What kind og loss to use",
        default="isotropic",
        choices=["diag_NIG", "isotropic", "KL_GMM_2"],
    )
    parser.add_argument(
        "--imbalanced",
        action="store_true",
    )
    parser.add_argument(
        "--ignore_subclusters",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--log_metrics_at_train",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--gpus",
        default=None
    )
    return parser

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(row_ind)):
            if row_ind[j] == y_pred[i]:
                best_fit.append(col_ind[j])
    return best_fit, row_ind, col_ind, w

def cluster_acc(y_true, y_pred):
    best_fit, row_ind, col_ind, w = best_cluster_fit(y_true, y_pred)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def train_cluster_net():
    parser = argparse.ArgumentParser(description="Only_for_embbedding")
    parser = parse_minimal_args(parser)
    parser = run_on_embeddings_hyperparams(parser)
    args = parser.parse_args()

    args.train_cluster_net = args.max_epochs
    args.features_dim = args.latent_dim
    
    if args.dataset == "synthetic":
        dataset_obj = GMM_dataset(args)
    else:
        dataset_obj = embbededDataset(args)
    train_loader, val_loader = dataset_obj.get_loaders()

    tags = ['umap_embbeded_dataset']
    if args.imbalanced:
        tags.append("unbalanced_trainset")

    if args.offline:
        logger = DummyLogger()
    else:
        logger = NeptuneLogger(
                api_key='your_API_token',
                project_name='your_project_name',
                experiment_name=args.exp_name,
                params=vars(args),
                tags=tags
            )

    if isinstance(logger, NeptuneLogger):
        if logger.api_key == 'your_API_token':
            print("No Neptune API token defined!")
            print("Please define Neptune API token or run with the --offline argument.")
            print("Running without logging...")
            logger = DummyLogger()

    # Main body
    if args.seed:
        pl.utilities.seed.seed_everything(args.seed)
    
    model = ClusterNetModel(hparams=args, input_dim=args.latent_dim, init_k=args.init_k)
    if args.save_checkpoints:
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(dirpath = f"./saved_models/{args.dataset}/{args.exp_name}")
        if not os.path.exists(f'./saved_models/{args.dataset}'):
            os.makedirs(f'./saved_models/{args.dataset}')
        os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}')
    else:
        checkpoint_callback = False
    
    trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, gpus=args.gpus, num_sanity_val_steps=0, checkpoint_callback=checkpoint_callback, limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)
    trainer.fit(model, train_loader, val_loader)

    # evaluate last model
    dataset = dataset_obj.get_train_data()
    data, labels = dataset.tensors[0], dataset.tensors[1].numpy()
    net_pred = model(data).argmax(axis=1).cpu().numpy()

    acc = np.round(cluster_acc(labels, net_pred), 5)
    nmi = np.round(NMI(net_pred, labels), 5)
    ari = np.round(ARI(net_pred, labels), 5)

    print(f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(net_pred))}")


if __name__ == "__main__":
    train_cluster_net()
