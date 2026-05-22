"""
Script de vérification de l'environnement CINR-repro.
Lance : python scripts/check_env.py
"""

import sys


def check_torch_mps() -> bool:
    """Vérifie que torch est installé et que MPS est disponible."""
    try:
        import torch

        print(f"torch version : {torch.__version__}")
        mps_available = torch.backends.mps.is_available()
        print(f"MPS available : {mps_available}")

        if mps_available:
            # Test rapide d'un tenseur sur MPS
            x = torch.randn(2, 3, device="mps")
            y = x * 2 + 1
            y_cpu = y.cpu()
            print(f"Test MPS OK  : shape={y_cpu.shape}, device mps→cpu OK")
        else:
            print("WARNING : MPS non disponible, fallback sur CPU")
            x = torch.randn(2, 3, device="cpu")
            y = x * 2 + 1
            print(f"Test CPU OK  : shape={y.shape}")

        return mps_available
    except ImportError:
        print("ERREUR : torch non installé")
        return False


def check_openslide() -> bool:
    """Vérifie que openslide est importable."""
    try:
        import openslide

        print(f"openslide OK  : {openslide.__version__}")
        return True
    except ImportError:
        print("ERREUR : openslide-python non installé")
        return False
    except Exception as e:
        print(f"ERREUR openslide : {e}")
        print("  Vérifie que 'brew install openslide' a été fait")
        return False


def main() -> None:
    """Lance toutes les vérifications."""
    print("=" * 50)
    print("CINR-repro — Environment Check")
    print("=" * 50)

    checks = [
        ("torch + MPS", check_torch_mps),
        ("openslide", check_openslide),
    ]

    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"ERREUR inattendue : {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    all_ok = all(r[1] for r in results)
    for name, ok in results:
        status = "✓ OK" if ok else "✗ FAIL"
        print(f"  {status} : {name}")

    print("=" * 50)
    if all_ok:
        print("Tous les checks OK — environnement prêt.")
    else:
        print("ATTENTION : certains checks ont échoué.")
        sys.exit(1)


if __name__ == "__main__":
    main()
