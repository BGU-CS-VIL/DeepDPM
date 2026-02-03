"""Autoencoder for optional feature learning."""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Simple autoencoder for feature learning.

    Architecture: encoder -> latent -> decoder
    Used for end-to-end feature learning with DeepDPM clustering.
    """

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """Initialize autoencoder.

        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions (e.g., [500, 500, 2000])
            latent_dim: Dimension of latent space
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: (N, input_dim) input data

        Returns:
            (N, latent_dim) latent representations
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction.

        Args:
            z: (N, latent_dim) latent representations

        Returns:
            (N, input_dim) reconstructed data
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through autoencoder.

        Args:
            x: (N, input_dim) input data

        Returns:
            Tuple of (latent, reconstruction)
                latent: (N, latent_dim)
                reconstruction: (N, input_dim)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction
