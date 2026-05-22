"""
Modèle CINR adapté pour Apple Silicon (MPS).

Basé sur l'article sur: Lee et al., "Convolutional Implicit Neural Representation
of pathology whole-slide images", MICCAI 2024.

Adaptations vs l'original :
- tinycudann (HashGrid encoding) -> Fourier Features (PyTorch natif)
- .cuda() -> .to(DEVICE) avec DEVICE = "mps"
- torch.cuda.synchronize() -> supprimé
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def get_device() -> str:
    """Retourne le device disponible : mps, cuda, ou cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class FourierFeatures(nn.Module):
    """
    Positional encoding par Fourier features (Tancik et al., NeurIPS 2020).

    Remplace le HashGrid encoding (tinycudann, CUDA-only) de l'original CINR.

    Args:
        input_dim: Dimension d'entrée (2 pour coordonnées x,y).
        mapping_size: Nombre de features de sortie (doit être pair).
        sigma: Écart-type de la matrice B aléatoire.
    """

    def __init__(
        self,
        input_dim: int = 2,
        mapping_size: int = 256,
        sigma: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.sigma = sigma

        # Matrice B aléatoire non entraînable
        self.register_buffer(
            "B",
            torch.randn(mapping_size, input_dim) * sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) coordonnées normalisées [0,1] ou [-1,1].

        Returns:
            (..., mapping_size * 2) features encodées.
        """
        x_proj = 2.0 * math.pi * (x @ self.B.t())
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class Sine(nn.Module):
    """Activation sinus avec fréquence ajustable."""

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SineCNN(nn.Module):
    """
    Module CINR : Conv2d avec activation Sine.

    Inspiré de la classe SineCNN du notebook original.
    Utilise Conv2d 3x3 (pas de Linear) pour préserver la structure spatiale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        # Conv2d 3×3 avec padding pour garder la même résolution spatiale
        self.cnn = nn.Conv2d(
            in_features, out_features,
            kernel_size=3, stride=1, padding=1, bias=True,
        )

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                nn.init.uniform_(
                    self.cnn.weight,
                    -1.0 / self.in_features,
                    1.0 / self.in_features,
                )
            else:
                nn.init.uniform_(
                    self.cnn.weight,
                    -math.sqrt(6.0 / self.in_features) / self.omega_0,
                    math.sqrt(6.0 / self.in_features) / self.omega_0,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Sine(w0=self.omega_0)(self.cnn(x))


class CINRModel(nn.Module):
    """
    CINR : Convolutional Implicit Neural Representation.

    Architecture :
    1. Coordonnées 2D -> Fourier Features (encodage positionnel)
    2. n couches de SineCNN (Conv2d 3x3 + activation Sine)
    3. Couche finale Linear -> RGB

    Args:
        fourier_mapping: Taille des Fourier features (défaut: 256 -> 512 après concat).
        hidden_features: Nombre de canaux dans les couches cachées.
        hidden_layers: Nombre de couches cachées SineCNN.
        omega_0: Fréquence de l'activation Sine (w0=30 pour la 1ere couche).
        output_channels: Nombre de canaux de sortie (3 pour RGB).
    """

    def __init__(
        self,
        fourier_mapping: int = 256,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        output_channels: int = 3,
    ) -> None:
        super().__init__()

        self.fourier = FourierFeatures(
            input_dim=2,
            mapping_size=fourier_mapping,
            sigma=4.0,
        )
        # FourierFeatures output = mapping_size * 2
        ff_out = fourier_mapping * 2

        layers: list[nn.Module] = []

        # Première couche : is_first=True
        layers.append(
            SineCNN(ff_out, hidden_features, is_first=True, omega_0=omega_0)
        )

        # Couches cachées
        for _ in range(hidden_layers - 1):
            layers.append(
                SineCNN(
                    hidden_features, hidden_features,
                    is_first=False, omega_0=omega_0,
                )
            )

        # Couche finale : Linear sur chaque pixel (Conv2d 1×1)
        layers.append(nn.Conv2d(hidden_features, output_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, H, W, 2) grille de coordonnées normalisées [0,1].

        Returns:
            (B, output_channels, H, W) valeurs RGB reconstruites.
        """
        # Encodage Fourier : (B, H, W, 2) → (B, H, W, F)
        encoded = self.fourier(coords)

        # Passage en format Conv2d : (B, H, W, F) → (B, F, H, W)
        encoded = encoded.permute(0, 3, 1, 2)

        # SineCNN + couche finale
        return self.net(encoded)  # (B, 3, H, W)
