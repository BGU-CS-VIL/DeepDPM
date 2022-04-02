#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


class embbededDataset:

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    def __init__(self, args):
        self.args = args
        self._input_dim = args.features_dim
        self._latent_dim = args.latent_dim,
        self.transform = args.transform
        self.dataset_loc = os.path.join(args.dir, f"{args.dataset.upper()}")

    def get_train_data(self):
        train_codes = torch.Tensor(torch.load(os.path.join(self.dataset_loc, "train_codes.pt")))
        labels = torch.load(os.path.join(self.dataset_loc, "train_labels.pt"))
        
        train_labels = torch.Tensor(labels).cpu().float()

        if self.transform:
            if self.transform == "standard":
                train_codes = torch.Tensor(StandardScaler().fit_transform(train_codes))
            elif self.transform in ["normalize", "standard_normalize"]:
                train_codes = torch.Tensor(Normalizer().fit_transform(train_codes))
            elif self.transform == "min_max":
                train_codes = torch.Tensor(MinMaxScaler().fit_transform(train_codes))
        train_set = TensorDataset(train_codes, train_labels)
        del train_codes
        del train_labels
        return train_set

    def get_test_data(self):
        if "N2D" in self.args.dataset:
            # Training and evaluating the entire dataset.
            # Take only a few examples just to not break code.
            data = self.get_train_data()
            val_codes = data.tensors[0][5:]
            val_labels = data.tensors[1][5:]
        else:
            val_codes = torch.Tensor(torch.load(os.path.join(self.dataset_loc, "val_codes.pt")))
            val_labels = torch.Tensor(torch.load(os.path.join(self.dataset_loc, "val_labels.pt")).cpu().float())
        if self.transform:
            if self.transform == "normalize":
                val_codes = torch.Tensor(Normalizer().fit_transform(val_codes))
            elif self.transform == "min_max":
                val_codes = torch.Tensor(MinMaxScaler().fit_transform(val_codes))
            elif self.transform == "standard":
                val_codes = torch.Tensor(StandardScaler().fit_transform(val_codes))
        test_set = TensorDataset(val_codes, val_labels)
        del val_codes
        del val_labels
        return test_set

    def get_train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            self.get_train_data(),
            batch_size=self.args.batch_size, shuffle=True, num_workers=3
        )
        return train_loader

    def get_test_loader(self):
        test_loader = torch.utils.data.DataLoader(
            self.get_test_data(),
            batch_size=self.args.batch_size, shuffle=False, num_workers=3
        )
        return test_loader

    def get_loaders(self):
        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()
        return train_loader, test_loader
