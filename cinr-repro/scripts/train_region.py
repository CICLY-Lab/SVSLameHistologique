"""
Entraîne un modèle CINR sur une région et sauvegarde le checkpoint.

Usage:
    python scripts/train_region.py \
        --region data/regions/region_000.png \
        --out outputs/regions/ \
        --target-psnr 30 \
        --max-epochs 15000 \
        --device cuda
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.cinr_model import CINRModel


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_coord_grid(h: int, w: int, device: torch.device
                    ) -> torch.Tensor:
    xs = torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=device)
    ys = torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xv, yv], dim=-1).unsqueeze(0)


def sample_random_patch(coords: torch.Tensor, image: torch.Tensor,
                        patch_size: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    _, H, W, _ = coords.shape
    y = random.randint(0, H - patch_size)
    x = random.randint(0, W - patch_size)
    return (
        coords[:, y:y + patch_size, x:x + patch_size, :],
        image[:, y:y + patch_size, x:x + patch_size, :],
    )


def predict_tiled(model: torch.nn.Module, coords: torch.Tensor,
                  tile_size: int = 256) -> torch.Tensor:
    """Évalue le modèle par tuiles pour éviter l'OOM sur grande image."""
    _, H, W, _ = coords.shape
    result = torch.zeros(1, 3, H, W, device=coords.device)
    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            patch = model(coords[:, y0:y1, x0:x1, :])
            result[:, :, y0:y1, x0:x1] = patch
    return result


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train_region(
    region_path: Path,
    out_dir: Path,
    target_psnr: float = 30.0,
    max_epochs: int = 15000,
    patch_size: int = 256,
    hidden_features: int = 256,
    fourier_mapping: int = 512,
    hidden_layers: int = 5,
    eval_tile_size: int = 256,
    lr: float = 1e-4,
    seed: int = 42,
) -> tuple[Path, float, float]:
    """Entraîne CINR jusqu'au PSNR cible ou max_epochs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Charger l'image ---
    img_np = np.array(Image.open(region_path).convert("RGB")).astype(np.float32) / 255.0
    H, W, C = img_np.shape
    img = torch.from_numpy(img_np).float().unsqueeze(0).to(device)

    print(f"Région     : {region_path.name}  ({W}×{H})")
    print(f"Device     : {device}")
    print(f"PSNR cible : {target_psnr:.1f} dB")
    print(f"Max epochs : {max_epochs}")
    print(f"Patch size : {patch_size}")

    # --- Grille de coordonnées ---
    full_coords = make_coord_grid(H, W, device)

    # --- Modèle ---
    model = CINRModel(
        fourier_mapping=fourier_mapping,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        omega_0=30.0,
        output_channels=C,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.001
    )
    loss_fn = torch.nn.MSELoss()

    # --- Boucle ---
    best_psnr = 0.0
    t_start = time.time()

    for step in range(1, max_epochs + 1):
        coords_patch, img_patch = sample_random_patch(
            full_coords, img, patch_size
        )
        target = img_patch.permute(0, 3, 1, 2)
        pred = model(coords_patch)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Évaluation périodique
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                full_pred = predict_tiled(model, full_coords, eval_tile_size)
                rec_np = full_pred[0].permute(1, 2, 0).cpu().numpy()
                rec_np = np.clip(rec_np, 0, 1)
                mse = np.mean((img_np - rec_np) ** 2)
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
                elapsed = time.time() - t_start

            if psnr > best_psnr:
                best_psnr = psnr

            print(f"  [{step:6d}] loss={loss.item():.6f} "
                  f"PSNR={psnr:.2f} dB  t={elapsed:.0f}s  best={best_psnr:.2f}")

            # Arrêt anticipé si PSNR atteint
            if psnr >= target_psnr:
                print(f"\n✓ PSNR cible {target_psnr:.1f} dB atteint en "
                      f"{elapsed:.0f}s ({step} epochs)")
                break

    t_total = time.time() - t_start

    # --- Sauvegarde ---
    stem = region_path.stem
    model_path = out_dir / f"{stem}_PSNR{best_psnr:.0f}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "psnr": best_psnr,
        "params": n_params,
        "epochs": step,
        "image_shape": (H, W, C),
        "hidden_features": hidden_features,
        "fourier_mapping": fourier_mapping,
        "hidden_layers": hidden_layers,
    }, model_path)

    model_size_kb = model_path.stat().st_size / 1024
    print(f"\n✓ Modèle sauvegardé : {model_path.name} "
          f"({model_size_kb:.0f} Ko)  PSNR={best_psnr:.2f} dB  "
          f"temps={t_total:.0f}s")

    return model_path, best_psnr, t_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîner CINR sur une région WSI"
    )
    parser.add_argument(
        "--region", type=str, required=True,
        help="Chemin PNG de la région"
    )
    parser.add_argument(
        "--out", type=str, default="outputs/regions/",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--target-psnr", type=float, default=30.0,
        help="PSNR cible (arrêt anticipé)"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=15000,
        help="Nombre max d'itérations"
    )
    parser.add_argument(
        "--patch-size", type=int, default=256,
        help="Taille du patch d'entraînement"
    )
    parser.add_argument(
        "--hidden-features", type=int, default=256,
        help="Canaux cachés CINR"
    )
    parser.add_argument(
        "--fourier-mapping", type=int, default=512,
        help="Taille des Fourier features"
    )
    parser.add_argument(
        "--hidden-layers", type=int, default=5,
        help="Nombre de couches SineCNN"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed aléatoire"
    )
    args = parser.parse_args()

    train_region(
        region_path=Path(args.region),
        out_dir=Path(args.out),
        target_psnr=args.target_psnr,
        max_epochs=args.max_epochs,
        patch_size=args.patch_size,
        hidden_features=args.hidden_features,
        fourier_mapping=args.fourier_mapping,
        hidden_layers=args.hidden_layers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
