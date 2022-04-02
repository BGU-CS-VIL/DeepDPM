import argparse
import torch
import umap
import numpy as np

from src.datasets import MNIST, CIFAR10, USPS
from src.embbeded_datasets import embbededDataset
from src.AE_ClusterPipeline import AE_ClusterPipeline


def parse_minimal_args(parser):
    # Dataset parameters
    parser.add_argument("--dir", default="/path/to/datasets", help="datasets directory")
    parser.add_argument("--dataset", default="mnist", help="the dataset used")

    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--ae_pretrain_path", type=str, default="/path/to/ae/weights", help="the path to pretrained ae weights"
    )
    parser.add_argument(
        "--umap_dim", type=int, default=10
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--imbalanced", type=bool, default=False
    )
    return parser

def make_embbedings():
    parser = argparse.ArgumentParser(description="Only_for_embbedding")
    parser = parse_minimal_args(parser)
    parser = AE_ClusterPipeline.add_model_specific_args(parser)

    args = parser.parse_args()

    # Load data
    if args.dataset == "mnist":
        data = MNIST(args)
    elif args.dataset == "cifar10":
        data = CIFAR10(args)
    elif args.dataset == "usps":
        data = USPS(args)
    else:
        data = None

    train_loader, val_loader = data.get_loaders(args)
    args.input_dim = data.input_dim

    # Main body
    ae = AE_ClusterPipeline(args=args, logger=None)
    ae.load_state_dict(torch.load(args.ae_pretrain_path))
    ae.eval()
    ae.freeze()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus is not None else "cpu")
    ae.to(device)

    train_codes, train_labels = [], []
    val_codes, val_labels = [], []

    for epoch in range(1):
        for i, data in enumerate(train_loader, 0):
            with torch.no_grad():
                inputs, labels = data[0].to(device), data[1].to(device)
                if args.dataset == "mnist":
                    inputs = inputs.view(inputs.size()[0], -1)
                codes = torch.from_numpy(ae.forward(inputs, latent=True))
                train_codes.append(codes.view(codes.shape[0], -1))
                train_labels.append(labels)

        train_codes = torch.cat(train_codes)
        train_labels = torch.cat(train_labels)

        # apply UMAP
        umap_obj = umap.UMAP(n_neighbors=20, min_dist=0, n_components=args.umap_dim)
        print("Starting umap for train...")
        train_codes = umap_obj.fit_transform(train_codes)
        print("Finishing umap...")

        if args.imbalanced:
            torch.seed()
            # sample different number of examples from each class
            if args.dataset in ("MNIST_N2D", "USPS_N2D"):
                imbalanced_classes_labels = torch.tensor([8, 2, 5, 9])
                count_vec = [0.1, 0.05, 0.2, 0.3]
            elif args.dataset == "FASHION_N2D":
                imbalanced_classes_labels = torch.tensor([0, 5, 7, 8, 3])
                count_vec = [0.37, 0.42, 0.54, 0.19, 0.19]

            mask = torch.ones_like(train_labels, dtype=bool)
            for per, gt_label in enumerate(imbalanced_classes_labels):
                indices_of_gt_label = (train_labels == gt_label).nonzero(as_tuple=True)[0]
                indices_to_remove = indices_of_gt_label[: int(np.floor(len(indices_of_gt_label)) * (1 - count_vec[per]))]
                mask.scatter_(0, indices_to_remove, 0)

            train_codes = train_codes[mask, :]
            train_labels = train_labels[mask]
        
        torch.save(train_codes, f'train_codes_{args.umap_dim}.pt')
        torch.save(train_labels, f'train_labels_{args.umap_dim}.pt')

        del train_labels
        del train_codes

        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data[0].to(device), data[1].to(device)
                if args.dataset == "mnist":
                    inputs = inputs.view(inputs.size()[0], -1)
                codes = torch.from_numpy(ae.forward(inputs, latent=True))
                val_codes.append(codes.view(codes.shape[0], -1))
                val_labels.append(labels)

        val_codes = torch.cat(val_codes)
        val_labels = torch.cat(val_labels)

        print("Starting umap for val...")
        val_codes = umap_obj.transform(val_codes)
        print("Finishing umap...")

        if args.imbalanced:
            torch.seed()
            # sample different number of examples from each class
            if args.dataset in ("MNIST_N2D"):
                imbalanced_classes_labels = torch.tensor([8, 2, 5, 9])
                count_vec = [0.1, 0.05, 0.2, 0.3]
            elif args.dataset in ("USPS_N2D"):
                imbalanced_classes_labels = torch.tensor([8, 2, 5, 9])
                count_vec = [0.1, 0.05, 0.2, 0.3]
            elif args.dataset == "FASHION_N2D":
                imbalanced_classes_labels = torch.tensor([0, 5, 7, 8, 3])
                count_vec = [0.37, 0.42, 0.54, 0.19, 0.19]

            mask = torch.ones_like(val_labels, dtype=bool)
            for per, gt_label in enumerate(imbalanced_classes_labels):
                indices_of_gt_label = (val_labels == gt_label).nonzero(as_tuple=True)[0]
                indices_to_remove = indices_of_gt_label[: int(np.floor(len(indices_of_gt_label)) * (1 - count_vec[per]))]
                mask.scatter_(0, indices_to_remove, 0)

            val_codes = val_codes[mask, :]
            val_labels = val_labels[mask]
        torch.save(val_codes, f'val_codes_{args.umap_dim}.pt')
        torch.save(val_labels, f'val_labels_{args.umap_dim}.pt')

        del val_labels
        del val_codes

if __name__ == "__main__":
    make_embbedings()