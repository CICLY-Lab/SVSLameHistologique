"""
Script d'entraînement CINR sur un patch histopathologique.

Usage :
    uv run python src/train_cinr.py --patch-idx 0 --epochs 5000

Charge un patch du dataset HuggingFace nathbns/SVS-TCGA-BR (256×256),
entraîne le modèle CINR, et sauvegarde la reconstruction dans outputs/.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cinr_model import CINRModel, get_device

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def load_patch(patch_idx: int = 0, min_tissue: float = 0.5) -> tuple[torch.Tensor, dict]:
    """
    Charge un patch du dataset HF avec assez de tissu.

    Args:
        patch_idx: Index dans le dataset trié par tissue_ratio décroissant.
        min_tissue: Ratio de tissu minimum requis.

    Returns:
        (image_tensor, meta) avec image_tensor shape (1, H, W, 3) normalisé [0,1].
    """
    from datasets import load_dataset

    ds = load_dataset("nathbns/SVS-TCGA-BR", split="train")
    # Filtrer les patches avec assez de tissu, trier par tissue_ratio décroissant
    ds_filtered = ds.filter(lambda x: x["tissue_ratio"] >= min_tissue)
    ds_sorted = ds_filtered.sort("tissue_ratio", reverse=True)

    if patch_idx >= len(ds_sorted):
        raise ValueError(
            f"patch_idx={patch_idx} >= {len(ds_sorted)} patches filtrés "
            f"(tissue_ratio >= {min_tissue})"
        )

    sample = ds_sorted[patch_idx]
    img_pil = sample["image"]
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).float()  # (H, W, 3)
    img_tensor = img_tensor.unsqueeze(0)  # (1, H, W, 3)

    meta = {
        "patch_id": sample.get("patch_id", patch_idx),
        "slide_name": sample.get("slide_name", "unknown"),
        "tissue_ratio": sample.get("tissue_ratio", 1.0),
        "x": sample.get("x", 0),
        "y": sample.get("y", 0),
    }
    return img_tensor, meta


def make_coord_grid(h: int, w: int, device: str) -> torch.Tensor:
    """
    Crée une grille de coordonnées normalisées [0, 1].

    Returns:
        (1, H, W, 2) tenseur.
    """
    xs = torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=device)
    ys = torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xv, yv], dim=-1)  # (H, W, 2)
    return coords.unsqueeze(0)  # (1, H, W, 2)


def sample_random_patch(
    coords: torch.Tensor,
    image: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Échantillonne un patch aléatoire de la grille de coordonnées et de l'image.

    Args:
        coords: (1, H, W, 2) grille complète.
        image: (1, H, W, 3) image complète.
        patch_size: Taille du patch carré.

    Returns:
        (coords_patch, image_patch) shapes (1, patch_size, patch_size, ...).
    """
    _, H, W, _ = coords.shape
    y = random.randint(0, H - patch_size)
    x = random.randint(0, W - patch_size)
    return (
        coords[:, y:y + patch_size, x:x + patch_size, :],
        image[:, y:y + patch_size, x:x + patch_size, :],
    )


def train(
    image_tensor: torch.Tensor,
    device: str,
    epochs: int = 5000,
    patch_size: int = 64,
    lr: float = 1e-4,
    log_interval: int = 500,
    output_dir: str = "outputs",
) -> dict:
    """
    Entraîne le modèle CINR sur un patch.

    Args:
        image_tensor: (1, H, W, 3) image normalisée [0,1].
        device: "mps" ou "cpu".
        epochs: Nombre d'itérations.
        patch_size: Taille des patches échantillonnés.
        lr: Learning rate.
        log_interval: Intervalle de logging (itérations).
        output_dir: Dossier de sortie.

    Returns:
        dict avec métriques finales.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Déplacer l'image sur le device
    img = image_tensor.to(device)
    _, H, W, C = img.shape

    # Auto-détection du patch_size optimal
    if patch_size is None:
        patch_size = min(256, H, W)
    print(f"Patch size : {patch_size}×{patch_size} (image {W}×{H})")

    # Grille de coordonnées complète
    full_coords = make_coord_grid(H, W, device)  # (1, H, W, 2)

    # Modèle
    model = CINRModel(
        fourier_mapping=256,
        hidden_features=128,
        hidden_layers=4,
        omega_0=30.0,
        output_channels=C,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : {n_params:,} paramètres")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    loss_fn = nn.MSELoss()

    best_psnr = 0.0
    psnr_log = []
    t_start = time.time()

    for step in range(1, epochs + 1):
        # Si patch_size >= taille image → entraîner sur l'image entière
        use_full = patch_size >= min(H, W)
        if use_full:
            coords_batch = full_coords
            target = img.permute(0, 3, 1, 2)
        else:
            coords_batch, img_patch = sample_random_patch(
                full_coords, img, patch_size,
            )
            target = img_patch.permute(0, 3, 1, 2)

        # Forward
        pred = model(coords_batch)
        loss = loss_fn(pred, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if step % log_interval == 0 or step == 1 or step == epochs:
            with torch.no_grad():
                # Pour les grandes images, évaluer le PSNR sur un patch
                if use_full or max(H, W) <= 512:
                    full_pred = model(full_coords)
                    orig_ref = img[0].cpu().numpy()
                    rec_ref = full_pred[0].permute(1, 2, 0).cpu().numpy()
                else:
                    coords_patch, img_patch = sample_random_patch(
                        full_coords, img, min(patch_size, 256),
                    )
                    full_pred = model(coords_patch)
                    orig_ref = img_patch[0].cpu().numpy()
                    rec_ref = full_pred[0].permute(1, 2, 0).cpu().numpy()
                rec_ref = np.clip(rec_ref, 0, 1)
                mse = np.mean((orig_ref - rec_ref) ** 2)
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
                psnr_log.append((step, psnr))
                elapsed = time.time() - t_start
                print(
                    f"[{step:5d}/{epochs}] loss={loss.item():.6f} "
                    f"PSNR={psnr:.2f} dB (best={best_psnr:.2f}) "
                    f"t={elapsed:.0f}s"
                )
                best_psnr = max(best_psnr, psnr)

    # Sauvegarde reconstruction finale (traitée par patchs si image trop grande)
    with torch.no_grad():
        if max(H, W) <= 512:
            final_pred = model(full_coords)
        else:
            # Inférence par tuiles pour les grandes images
            tile_size = 256
            final_pred = torch.zeros(1, C, H, W, device=device)
            for y in range(0, H, tile_size):
                y_end = min(y + tile_size, H)
                for x in range(0, W, tile_size):
                    x_end = min(x + tile_size, W)
                    coords_tile = full_coords[:, y:y_end, x:x_end, :]
                    pred_tile = model(coords_tile)
                    final_pred[:, :, y:y_end, x:x_end] = pred_tile
        final_psnr = _compute_psnr(img, final_pred)

    _save_reconstruction(img, final_pred, out_dir)

    # Sauvegarde modèle
    torch.save(model.state_dict(), out_dir / "model.pth")
    print(f"\nModèle sauvegardé : {out_dir / 'model.pth'}")
    print(f"PSNR final : {final_psnr:.2f} dB")

    return {
        "psnr": final_psnr,
        "best_psnr": best_psnr,
        "n_params": n_params,
        "epochs": epochs,
        "elapsed_s": time.time() - t_start,
    }


def _compute_psnr(
    original: torch.Tensor, reconstruction: torch.Tensor,
) -> float:
    """Calcule le PSNR entre l'original et la reconstruction."""
    orig = original.permute(0, 3, 1, 2).cpu().numpy()
    rec = reconstruction.cpu().numpy()
    orig = np.clip(orig, 0.0, 1.0)
    rec = np.clip(rec, 0.0, 1.0)
    return float(psnr_fn(orig, rec, data_range=1.0))


def _save_reconstruction(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    out_dir: Path,
) -> None:
    """Sauvegarde l'image originale et la reconstruction en PNG."""
    orig_np = original[0].cpu().numpy()
    rec_np = reconstruction[0].permute(1, 2, 0).cpu().numpy()

    orig_np = np.clip(orig_np * 255, 0, 255).astype(np.uint8)
    rec_np = np.clip(rec_np * 255, 0, 255).astype(np.uint8)

    Image.fromarray(orig_np).save(out_dir / "original.png")
    Image.fromarray(rec_np).save(out_dir / "reconstruction.png")
    print(f"Images sauvegardées : {out_dir}/original.png, reconstruction.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement CINR sur un patch histopathologique"
    )
    parser.add_argument(
        "--patch-idx", type=int, default=None,
        help="Index du patch dans le dataset HF (trié par tissue_ratio décroissant)"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Chemin vers une image PNG/JPG locale (prioritaire sur --patch-idx)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Nombre d'itérations d'entraînement"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--patch-size", type=int, default=None,
        help="Taille des patches (défaut: auto = min(256, H, W))"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (mps, cpu). Défaut: auto-détection."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed aléatoire"
    )
    args = parser.parse_args()

    # Reproductibilité
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or get_device()
    print(f"Device : {device}")
    print(f"Patch size : {'auto' if args.patch_size is None else f'{args.patch_size}×{args.patch_size}'}")
    print(f"Epochs : {args.epochs}")

    # Charger le patch
    if args.image:
        # Charger depuis un fichier local
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image introuvable : {img_path}")
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0)
        meta = {"source": str(img_path), "size": img_pil.size}
        print(f"Image chargée : {img_tensor.shape} — {meta}")
    else:
        # Charger depuis le dataset HuggingFace
        patch_idx = args.patch_idx if args.patch_idx is not None else 0
        img_tensor, meta = load_patch(patch_idx=patch_idx)
        print(f"Patch chargé : {img_tensor.shape} — {meta}")

    # Entraîner
    metrics = train(
        image_tensor=img_tensor,
        device=device,
        epochs=args.epochs,
        patch_size=args.patch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 50)
    print("Résumé")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:15s} : {v}")


if __name__ == "__main__":
    main()
