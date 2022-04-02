#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
from collections import OrderedDict
import math


class AutoEncoder(nn.Module):
    def __init__(self, args, input_dim=None):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim or args.input_dim
        self.output_dim = self.input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(self.latent_dim)
        self.dims_list = (
            args.hidden_dims + args.hidden_dims[:-1][::-1]
        )  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.n_clusters = args.n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == args.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        "linear0": nn.Linear(self.input_dim, hidden_dim),
                        "activation0": nn.ReLU(),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(
                            self.hidden_dims[idx - 1], hidden_dim
                        ),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(self.hidden_dims[idx]),
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(hidden_dim, self.output_dim),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx + 1]
                        ),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx + 1]),
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = "[Structure]: {}-".format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += "{}-".format(dim)
        repr_str += str(self.output_dim) + "\n"
        repr_str += "[n_layers]: {}".format(self.n_layers) + "\n"
        repr_str += "[n_clusters]: {}".format(self.n_clusters) + "\n"
        repr_str += "[input_dims]: {}".format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)

    def decode(self, latent_X):
        return self.decoder(latent_X)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1).float()


class UnFlatten(torch.nn.Module):

    def __init__(self, channel, width) -> None:
        super().__init__()
        self.channel = channel
        self.width = width

    def forward(self, x):
        return x.reshape(-1, self.channel, self.width, self.width)


class ConvAutoEncoder(nn.Module):
    def __init__(self, args, input_dim=None):
        super(ConvAutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim or args.input_dim
        self.output_dim = self.input_dim
        self.latent_dim = args.latent_dim

        # encoder #
        self.encoder_conv = nn.Sequential(
            UnFlatten(channel=1, width=16),                       # [batch, 1, 16, 16]
            nn.Conv2d(1, 32, 5, stride=1),                       # [batch, 32, 12, 12]
            nn.BatchNorm2d(32),                                  # [batch, 32, 12, 12]
            nn.ReLU(),

        )
        self.encoder_maxPool = nn.MaxPool2d(2, stride=2, return_indices=True)  # [batch, 32, 6, 6]
        self.encoder_linear = nn.Sequential(
            Flatten(),                                           # [batch, 1152]
            nn.Linear(32 * 6 * 6, self.latent_dim)
        )

        # decoder #
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * 6 * 6),
            UnFlatten(channel=32, width=6),
        )                                                       # [batch, 32, 6, 6]
        self.decoder_maxPool = nn.MaxUnpool2d(2, stride=2)      # [batch, 32, 12, 12]
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 5, stride=1),              # [batch, 1, 16, 16]
            Flatten()
        )

    def forward(self, X, latent=False):
        output = self.encode(X)
        if latent:
            return output
        return self.decode(output)

    def encode(self, X):
        out = self.encoder_conv(X)
        out, self.ind = self.encoder_maxPool(out)
        return self.encoder_linear(out)

    def decode(self, X):
        out = self.decoder_linear(X)
        out = self.decoder_maxPool(out, self.ind)
        return self.decoder_conv(out)

