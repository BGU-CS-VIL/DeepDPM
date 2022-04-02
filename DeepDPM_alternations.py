#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
import numpy as np

from src.AE_ClusterPipeline import AE_ClusterPipeline
from src.datasets import MNIST, REUTERS
from src.embbeded_datasets import embbededDataset
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from DeepDPM import cluster_acc


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--dir", default="/path/to/dataset/", help="dataset directory")
    parser.add_argument("--dataset", default="mnist")

    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=0, help="number of pre-train epochs"
    )

    parser.add_argument(
        "--pretrain", action="store_true", help="whether use pre-training"
    )

    parser.add_argument(
        "--pretrain_path", type=str, default="./saved_models/ae_weights/mnist_e2e", help="use pretrained weights"
    )

    # Model parameters
    parser = AE_ClusterPipeline.add_model_specific_args(parser)
    parser = ClusterNetModel.add_model_specific_args(parser)

    # Utility parameters
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
        "--log-interval",
        type=int,
        default=400,
        help="how many batches to wait before logging the training status",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="short test run on a few instances of the dataset",
    )

    # Logger parameters
    parser.add_argument(
        "--tag",
        type=str,
        default="Replicate git results",
        help="Experiment name and tag",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )

    parser.add_argument(
        "--features_dim",
        type=int,
        default=128,
        help="features dim of embedded datasets",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="number of AE epochs",
    )
    parser.add_argument(
        "--number_of_ae_alternations",
        type=int,
        default=3,
        help="The number of DeepDPM AE alternations to perform"
    )
    parser.add_argument(
        "--save_checkpoints", type=bool, default=False
    )
    parser.add_argument(
        "--exp_name", type=str, default="default_exp"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run training without Neptune Logger"
    )
    parser.add_argument(
        "--gpus",
        default=None
    )
    args = parser.parse_args()
    return args


def load_pretrained(args, model):
    if args.pretrain_path is not None and args.pretrain_path != "None":
        # load ae weights
        state = torch.load(args.pretrain_path)
        new_state = {}
        for key in state.keys():
            if key[:11] == "autoencoder":
                new_state["feature_extractor." + key] = state[key]
            else:
                new_state[key] = state[key]

        model.load_state_dict(new_state)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    args.n_clusters = args.init_k

    if args.seed:
        pl.utilities.seed.seed_everything(args.seed)

    # Load data
    if args.dataset == "mnist":
        data = MNIST(args)
    elif args.dataset == "reuters10k":
        data = REUTERS(args, how_many=10000)
    else:
        # Used for ImageNet-50
        data = embbededDataset(args)

    train_loader, val_loader = data.get_loaders()
    args.input_dim = data.input_dim

    tags = ['DeepDPM with alternations']
    tags.append(args.tag)
    if args.offline:
        from pytorch_lightning.loggers.base import DummyLogger
        logger = DummyLogger()
    else:
        logger = NeptuneLogger(
                api_key='your_api_token',
                project_name='your_project_name',
                experiment_name=args.tag,
                params=vars(args),
                tags=tags
            )
    
    device = "cuda" if torch.cuda.is_available() and args.gpus is not None else "cpu"
    if isinstance(logger, NeptuneLogger):
        if logger.api_key == 'your_API_token':
            print("No Neptune API token defined!")
            print("Please define Neptune API token or run with the --offline argument.")
            print("Running without logging...")
            logger = DummyLogger()

    # Main body
    model = AE_ClusterPipeline(args=args, logger=logger)
    if not args.pretrain:
        load_pretrained(args, model)
    if args.save_checkpoints:
        if not os.path.exists(f'./saved_models/{args.dataset}'):
            os.makedirs(f'./saved_models/{args.dataset}')
        os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}')

    max_epochs = args.epoch * (args.number_of_ae_alternations - 1) + 1

    trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, gpus=args.gpus, num_sanity_val_steps=0, checkpoint_callback=False)
    trainer.fit(model, train_loader, val_loader)

    model.to(device=device)
    DeepDPM = model.clustering.model.cluster_model
    DeepDPM.to(device=device)
    # evaluate last model
    for i, dataset in enumerate([data.get_train_data(), data.get_test_data()]):
        data_, labels_ = dataset.tensors[0], dataset.tensors[1].numpy()
        pred = DeepDPM(data_.to(device=device)).argmax(axis=1).cpu().numpy()

        acc = np.round(cluster_acc(labels_, pred), 5)
        nmi = np.round(NMI(pred, labels_), 5)
        ari = np.round(ARI(pred, labels_), 5)
        if i == 0:
            print("Train evaluation:")
        else:
            print("Validation evaluation")
        print(f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(pred))}")
    model.cpu()
    DeepDPM.cpu()
