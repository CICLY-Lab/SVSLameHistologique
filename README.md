# Compression des lames histopathologiques SVS / DICOM

Projet de stage réalisé au laboratoire **CICLY**. Étude et expérimentation de
méthodes de compression pour les **Whole Slide Images (WSI)** utilisées en
anatomopathologie numérique, dont les fichiers atteignent couramment plusieurs
gigaoctets.

> **Objectif** : évaluer les codecs classiques et les approches neuronales
> (INR, compression apprise) pour réduire le volume de stockage des lames SVS
> tout en préservant la qualité diagnostique.

---

## Sommaire

- [Contexte](#contexte)
- [Arborescence du dépôt](#arborescence-du-dépôt)
- [Ce qui a été fait](#ce-qui-a-été-fait)
- [Données utilisées](#données-utilisées)
- [Pour reproduire](#pour-reproduire)
- [Documentation associée](#documentation-associée)

---

## Contexte

Les WSI au format **SVS** (Aperio/Leica) sont des conteneurs TIFF pyramidaux :
chaque niveau est tuilé (souvent 256×256 ou 512×512) et compressé en JPEG. Les
fichiers DICOM du labo suivent une logique similaire. Compresser avec ZIP/GZIP
ne donne presque rien car les tuiles sont **déjà compressées** : il faut donc
recompresser les tuiles, changer de codec, ou utiliser une approche neuronale.

---

## Arborescence du dépôt

```
.
├── data_extraction/          # Extraction de patches depuis les lames SVS (OpenSlide)
├── premiers_tests/           # Premiers essais exploratoires
├── expe_compression_classique/  # Benchmark JPEG / WebP / JPEG2000 / SVG
├── cinr-repro/               # Reproduction de CINR (Implicit Neural Representation)
│   ├── src/                  #   modèle + entraînement
│   ├── scripts/              #   lancement sur P2CHPD
│   └── notebooks/            #   démos visuelles
├── learned_compression/      # Benchmarks CompressAI (bmshj2018, mbt2018)
├── docs/                     # Site de présentation des résultats
├── dicom_analyse.ipynb       # Analyse des fichiers DICOM du labo
├── rapport_etat_art.md       # État de l'art détaillé (SOTA)
├── source_utiles.md          # Liens articles / repos utiles
├── guide_p2chpd.md           # Guide d'utilisation du cluster P2CHPD
├── requirements.txt          # Dépendances Python principales
└── svs_format.pdf            # Spécifications du format SVS
```

---

## Ce qui a été fait

### 1. État de l'art
Recensement des codecs classiques (JPEG, JPEG 2000, JPEG XL, WebP, AVIF), des
méthodes adaptatives (suppression de fond) et des approches neuronales
(INR/CINR, compression apprise/CLERIC). Voir [`rapport_etat_art.md`](rapport_etat_art.md).

### 2. Extraction de données
~30 000 tuiles 256×256 et 500 tuiles 2048×2048 extraites des zones tissulaires
de 91 lames SVS TCGA, puis publiées sur HuggingFace. Voir `data_extraction/`.

### 3. Benchmark des codecs classiques
Comparaison JPEG / WebP / JPEG 2000 sur les datasets extraits, avec métriques
BPP, PSNR et comparaisons visuelles. Voir `expe_compression_classique/`.
Un test de conversion SVG a été essayé mais abandonné (images trop complexes).

### 4. Reproduction de CINR
Implémentation d'un pipeline inspiré de **CINR** (Lee et al., MICCAI 2024) :
un réseau de neurones overfit sur une seule image pour la reconstruire à partir
de ses poids. Entraînement lancé sur le supercalculateur **P2CHPD** (A100).
Voir `cinr-repro/`.

### 5. Compression apprise (CompressAI)
Benchmark de deux modèles pré-entraînés via la lib **CompressAI** :
- `bmshj2018_factorized` (baseline sans hyperprior)
- `mbt2018_mean` (avec hyperprior — **meilleurs résultats obtenus**)

Tests sur régions 5000×5000 et 10000×10000. Voir `learned_compression/`.

### 6. Analyse des fichiers DICOM / MRXS du labo
Inspection des niveaux de compression et comparaison avec les formats SVS/MRXS.
Voir `dicom_analyse.ipynb`.

---

## Données utilisées

- **91 lames SVS** issues de TCGA-BRCA
- Datasets HuggingFace :
  - https://huggingface.co/datasets/nathbns/SVS-TCGA-BR (tuiles 256×256)
  - https://huggingface.co/datasets/nathbns/SVS-TCGA-2048 (tuiles 2048×2048)

---

## Pour reproduire

### Environnement local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> OpenSlide nécessite aussi l'installation système de `openslide` (Homebrew,
> apt, etc.) selon votre OS.

### Sur le cluster P2CHPD

Le fichier [`guide_p2chpd.md`](guide_p2chpd.md) détaille la connexion, la
création de l'environnement Python (dans `/tmp`, car `/home_nfs` est `noexec`),
et la soumission de jobs SLURM sur les nœuds GPU A100.

---

## Documentation associée

| Document | Contenu |
|---|---|
| [`rapport_etat_art.md`](rapport_etat_art.md) | État de l'art : codecs, INR, compression apprise |
| [`source_utiles.md`](source_utiles.md) | Liens articles / repos de référence |
| [`guide_p2chpd.md`](guide_p2chpd.md) | Guide d'utilisation du cluster P2CHPD |
| [`learned_compression/README.md`](learned_compression/README.md) | Détails des benchmarks CompressAI |
| [`README_AUTHOR.md`](README_AUTHOR.md) | Carnet de bord hebdomadaire du stage |

---

## Pistes pour la suite

- Fine-tuning d'un modèle de compression apprise sur les patches histologiques
- Évaluation à plus grande échelle (lame entière, pas seulement des régions)
- Comparaison avec CLERIC (Lee et al., 2025) si code disponible
- Mesure de l'impact diagnostic (évaluation par pathologistes)

---

## Auteur

Stage réalisé au laboratoire **CICLY**.
Code transféré au labo pour les futurs contributeurs — toute la documentation
interne est en français.
