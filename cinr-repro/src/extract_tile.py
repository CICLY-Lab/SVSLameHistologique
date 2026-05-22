"""
Extraction d'une tuile depuis une lame WSI (.svs/.ndpi/.tiff) via OpenSlide.

Usage:
    uv run python src/extract_tile.py \\
        /Users/nath/Desktop/data_stage/TCGA-xxx.svs \\
        --out data/tiles/tuile_test.png \\
        --level 0 --x 50000 --y 50000 --size 1024

Ou directement en Python:
    from src.extract_tile import extract_tile
    img = extract_tile("/path/to/wsi.svs", level=0, x=50000, y=50000, size=1024)
    img.save("data/tiles/tuile.png")

Fallback: si pas de WSI dispo, le script `scripts/generate_dummy_tile.py`
crée une image synthétique de test dans `data/raw/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def extract_tile(
    wsi_path: str | Path,
    level: int = 0,
    x: int = 0,
    y: int = 0,
    size: int = 1024,
) -> Image.Image:
    """
    Extrait une tuile carrée d'une WSI au niveau donné.

    Args:
        wsi_path: Chemin vers le fichier SVS/NDPI/TIFF.
        level: Niveau de pyramide OpenSlide (0 = pleine résolution).
        x: Coordonnée X du coin supérieur gauche (en pixels du niveau 0).
        y: Coordonnée Y du coin supérieur gauche (en pixels du niveau 0).
        size: Taille de la tuile en pixels (carré).

    Returns:
        PIL Image RGB de la tuile.

    Exemple:
        >>> img = extract_tile(
        ...     "/Users/nath/Desktop/data_stage/TCGA-VQ-A8E0-01Z-00-DX1.xxx.svs",
        ...     level=0, x=50000, y=50000, size=1024,
        ... )
        >>> img.save("data/tiles/tuile_1024.png")
    """
    import openslide

    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI introuvable : {wsi_path}")

    slide = openslide.OpenSlide(str(wsi_path))
    w0, h0 = slide.level_dimensions[0]
    downsample = int(slide.level_downsamples[level])

    # Ajuster les coordonnées si elles dépassent
    x = max(0, min(x, w0 - size * downsample))
    y = max(0, min(y, h0 - size * downsample))
    read_size = size * downsample

    print(f"WSI          : {wsi_path.name}")
    print(f"Dims niveau 0: {w0} x {h0}")
    print(f"Niveau       : {level} (downsample ×{downsample})")
    print(f"Extraction   : x={x}, y={y}, size={size}")

    # Lire au niveau 0, la taille demandée
    region = slide.read_region((x, y), level, (size, size)).convert("RGB")
    slide.close()
    return region


def generate_dummy_tile(
    out_path: str | Path,
    size: int = 512,
) -> Path:
    """
    Génère une image synthétique de test (cercles concentriques bruités).
    Utile pour tester l'entraînement sans WSI.

    Args:
        out_path: Chemin de sortie (PNG).
        size: Taille en pixels.

    Returns:
        Chemin du fichier créé.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)

    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i, freq in enumerate([3, 8, 20]):
        phase = np.random.rand() * 2 * np.pi
        channel = (np.sin(freq * r + phase) * 0.5 + 0.5)
        img[:, :, i] = (channel * 255).astype(np.uint8)

    # Ajouter du bruit de texture
    noise = np.random.randn(size, size, 3) * 30
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="RGB")
    pil_img.save(str(out_path), format="PNG")
    print(f"Image dummy générée : {out_path} ({size}×{size})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction d'une tuile WSI via OpenSlide"
    )
    parser.add_argument(
        "wsi", type=str, nargs="?",
        help="Chemin vers le fichier WSI (.svs/.ndpi/.tiff)"
    )
    parser.add_argument(
        "--out", type=str, default="data/tiles/tuile_1024.png",
        help="Chemin de sortie PNG (défaut: data/tiles/tuile_1024.png)"
    )
    parser.add_argument(
        "--level", type=int, default=0,
        help="Niveau de pyramide (défaut: 0)"
    )
    parser.add_argument(
        "--x", type=int, default=50000,
        help="Coordonnée X (défaut: 50000)"
    )
    parser.add_argument(
        "--y", type=int, default=50000,
        help="Coordonnée Y (défaut: 50000)"
    )
    parser.add_argument(
        "--size", type=int, default=1024,
        help="Taille de la tuile en pixels (défaut: 1024)"
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="Générer une image test au lieu d'extraire d'une WSI"
    )
    args = parser.parse_args()

    if args.dummy or args.wsi is None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_dummy_tile(out_path, size=args.size)
    else:
        tile = extract_tile(
            args.wsi,
            level=args.level,
            x=args.x,
            y=args.y,
            size=args.size,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tile.save(str(out_path), format="PNG")
        print(f"Tuile sauvegardée : {out_path} ({tile.size})")


if __name__ == "__main__":
    main()
