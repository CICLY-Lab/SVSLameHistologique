# learned_compression — Benchmarks CompressAI

Comparaison de méthodes de **compression neuronale** (apprise) avec JPEG comme référence, sur des régions extraites de lames SVS histopathologiques.

## Notebooks

| Fichier | Région | Modèles testés | Temps (A100) |
|---|---|---|---|
| [`benchmark_complet.ipynb`](benchmark_complet.ipynb) | 5000×5000 | JPEG, `bmshj2018_factorized` (q=1..8), `mbt2018_mean` (q=1..8) | ~15 min |
| [`benchmark_10_000.ipynb`](benchmark_10_000.ipynb) | 10000×10000 | idem | ~45 min |
| [`synthese_p2chpd.ipynb`](synthese_p2chpd.ipynb) | n/a (synthèse) | — | < 1 min |

## Modèles utilisés (via CompressAI)

- **`bmshj2018_factorized`** : autoencodeur simple, sans hyperprior. Référence "baseline" de la compression apprise.
- **`mbt2018_mean`** : avec hyperprior. Meilleure qualité à BPP comparable.

Poids pré-entraînés téléchargés automatiquement par `compressai.zoo`.

## Sortie clé

[`outputs/results_p2chpd.csv`](outputs/results_p2chpd.csv) — tableau final comparatif (BPP, PSNR, taille).

## Images

Toutes les images de comparaison (`region_*.png`, `mbt2018_*.png`, etc.) sont dans `docs/static/images/learned_compression/`. Elles sont **non versionnées** (trop lourdes, 30-80 Mo chacune) — voir [`ASSETS.md`](../ASSETS.md) pour la regénération.

## Pour relancer

```python
from compressai.zoo import bmshj2018_factorized, mbt2018_mean
m1 = bmshj2018_factorized(quality=5, pretrained=True).eval()
m2 = mbt2018_mean(quality=5, pretrained=True).eval()
```

Image de test : région 5000×5000 extraite d'une lame SVS via OpenSlide (cf. `data_extraction/`).
