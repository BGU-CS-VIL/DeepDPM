#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
import torch.nn as nn
import argparse
from src import datasets
from tqdm import tqdm
from src.get_embbedings.imagenet import ImageNetSubset, ImageNet


data_to_class_dict = {
    "MNIST": datasets.MNIST,
    "MNIST_TEST": datasets.MNIST_TEST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100-20": datasets.CIFAR100_20,
    "CIFAR20": datasets.CIFAR100_20,
    "STL10": datasets.STL10,
    "STL10_unlabeled_train": datasets.STL10,
    "imagenet_50": ImageNetSubset(subset_file="./src/get_embbedings/imagenet_subsets/imagenet_50.txt"),
    "imagenet_50_test": ImageNetSubset(subset_file="./src/get_embbedings/imagenet_subsets/imagenet_50.txt", split='test'),
    "imagenet": ImageNet()
    }


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--dir", default="/path/to/Datasets", help="datasets directory")
    parser.add_argument("--dataset", default="mnist", help="current dataset")

    # Pretrained weights parameters
    parser.add_argument("--pretrain_path", default='/path/to/pretrained/weights.pth.tar', help="pretrained weights path")

    # Feature extraction parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_extractor", type=str, default="simclr", choices=["simclr", "moco", "scan_simclr"])
    parser.add_argument("--outdir", type=str, default='./embeddings_results', help="location to save the pretrained embeddings")
    parser.add_argument("--features_dim", type=int, default=128, help="The resulting embbedings dim")

    args = parser.parse_args()
    return args


def load_feature_extractor(args):
    # Load backbofne
    if "simclr" in args.feature_extractor:
        from models.resnet_cifar import resnet18
        backbone = resnet18()

    elif "moco" in args.feature_extractor:
        from models.resnet import resnet50
        backbone = resnet50()  

    # Load model and pretrained weights
    if args.feature_extractor in ('simclr', 'moco'):
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone)

    state = torch.load(args.pretrain_path, map_location='cpu')

    if args.feature_extractor == "moco":
        new_state_dict = {}
        state = state['state_dict']
        for k in list(state.keys()):
            # Copy backbone weights
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                new_k = 'backbone.' + k[len('module.encoder_q.'):]
                new_state_dict[new_k] = state[k]

            # Copy mlp weights
            elif k.startswith('module.encoder_q.fc'):
                new_k = 'contrastive_head.' + k[len('module.encoder_q.fc.'):] 
                new_state_dict[new_k] = state[k]

            else:
                raise ValueError('Unexpected key {}'.format(k))
        state = new_state_dict

    missing = model.load_state_dict(state, strict=False)
    print("Finished loading weights.")
    print(f"Mismatched keys: {missing}")
    return model


def load_data(args):
    if "imagenet" in args.dataset:
        train_loader = data_to_class_dict[args.dataset].get_loader()
        test_loader = data_to_class_dict[args.dataset+"_test"].get_loader()
    else:
        if "unlabeled_train" in args.dataset:
            dataset = data_to_class_dict[args.dataset](args, split="train+unlabeled")

        else:
            dataset = data_to_class_dict[args.dataset](args)
        train_loader, test_loader = dataset.get_loaders()
    return train_loader, test_loader


def extract_features(args, model, train_loader, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus is not None else "cpu")
    model.to(device=device)
    train_codes, train_labels = [], []
    test_codes, test_labels = [], []

    for i, data in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)
            codes = model(inputs)
            train_codes.append(codes.view(codes.shape[0], -1))
            train_labels.append(labels)

    train_codes = torch.cat(train_codes).cpu()
    train_labels = torch.cat(train_labels).cpu()

    D = train_codes.shape[1]

    # if path does not exist, create it
    save_location = os.path.join(args.outdir, args.feature_extractor.upper(), args.dataset.upper()+ f"_{D}D")
    from pathlib import Path
    Path(save_location).mkdir(parents=True, exist_ok=True)

    print("Saving train embeddings...")
    print(f"train codes dims = {train_codes.shape}")
    D = train_codes.shape[1]
    # if path does not exist, create it
    save_location = os.path.join(args.outdir, args.feature_extractor.upper(), args.dataset.upper()+ f"_{D}D")
    from pathlib import Path
    Path(save_location).mkdir(parents=True, exist_ok=True)
    torch.save(train_codes, os.path.join(save_location, "train_codes.pt"))
    torch.save(train_labels, os.path.join(save_location, "train_labels.pt"))
    print("Saved train embeddings!")
    del train_codes, train_labels

    for i, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)
            codes = model(inputs)
            test_codes.append(codes.view(codes.shape[0], -1))
            test_labels.append(labels)

    test_codes = torch.cat(test_codes).cpu()
    test_labels = torch.cat(test_labels).cpu()

    print("Saving test embeddings...")
    print(f"test codes dims = {test_codes.shape}")
    torch.save(test_codes, os.path.join(save_location, "test_codes.pt"))
    torch.save(test_labels, os.path.join(save_location, "test_labels.pt"))
    print("Saved test embeddings!")

def main():
    args = parse_args()
    model = load_feature_extractor(args)
    train_loader, test_loader = load_data(args)
    extract_features(args, model, train_loader, test_loader)


if __name__ == "__main__":
    main()
