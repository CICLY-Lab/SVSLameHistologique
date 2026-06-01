# Compression d'Images SVS/DICOM

### Rapport détaillé du SOTA + source utiles sur les liens ci-dessous:
[liens vers SOTA](rapport_etat_art.md) <br/>
[liens vers source utile](source_utiles.md)

## Contexte
Les Whole Slide Images (WSI) utilisées en anatomopathologie numérique atteignent 
des résolutions extrêmes, générant des fichiers 
de plusieurs gigaoctets. 

### Carnet de bords à la semaine (mise à jours chaque fin de semaine):

**Semaine 1**:
- Phase de compréhension du sujet, des problématiques et plus précisément du fonctionnement des images WSI (Whole Slide Image)
- Recherche de l'état de l'art, codec utilisé pour les formats de compression, le SOTA actuelle, et ce qui est encore à l'état de recherche
- Premières expérimentation, en réalisant un benchmark des formats de compression les plus simples. Afin de me rendre compte des différences et des faiblesses de chaque méthodes de compression basique.

**Semaine 2**:
- Traitement des données, j'ai récupéré ~30 000 tuiles 256×256 extraites des zones tissulaires que j'ai ensuite upload sur huggingFace [lien du dataset 1](https://huggingface.co/datasets/nathbns/SVS-TCGA-BR)
- Traitement des données une nouvelle fois mais autre format plus intéressant, j'ai récupéré 500  tuiles 2048x2048 extraites des zones tissulaires que j'ai ensuite upload sur huggingFace [lien du dataset 2](https://huggingface.co/datasets/nathbns/SVS-TCGA-2048)
- Continuer le benchmark, affinage des résultats, mise en place de métrique d'évaluations de nos résultats. [lien benchmark](expe_compression_classique/benchmarkJPEG_WEBP_JPEG2000.ipynb)

**Semaine 3**:
- Recherche d'une manière de tokenizer l'image sous forme vectoriel: 
    - svg: Pas possible car image trop complexe. On ne peut pas représenter avec les courbes de bezier ou divers géométrie de manière simple.
- Implémentation d'un pipeline proche de **CINR**. Pour commencer, je vais essayer sur une image, l'obj est le suivant, prendre un fichier .svs, entrainer un modèle de reseau de neurone (comme CINR) afin de overfit le modèle sur une image pour qu'il puisse la reproduire à partir des poids du modèle. Commencement de teste d'entrainement sur le supercalculateur.

**Semaine 4**:
- Terminer implémentation du **CINR** puis tester sur une petite image
- Présentation slide + début rapport
- Exploration de nouvelles méthode de conversions vers SVG

**Semaine 5** :
- Analyser les fichier DICOM du labo (nv de compression, etc..)
- MRXS (comparer avec le svs/dicom)
- CINR sur un plus grand fichier 
- Benchmark de deux modèle de compression neuronale (type CLERIC):
    - `bmshj2018_factorized` et `mbt2018_mean` via la lib compressai
    - Pour l'instant ce sont nos meilleurs résultats
    - Semble etre notre meilleure piste et le SOTA

**Semaine 6 (en cours)**
- Continuer nos analyses des méthodes de compression neuronale avec le MLCI++, Cheng2020, Ballé2018
- Voir le coût si appliqué à un TIFF complet:
    - une fois le modèle le plus performant trouvé -> Fine tuning?



### Travail effectué

- État de l'art: [lien SOTA](rapport_etat_art.md) 
- Analyse fichier SVS et première expérimentation (JPEG, WebP, JPEG2K, JPEG XL, AVIF): [lien benchmark](expe_compression_classique/01_analyse_et_premiere_expe.ipynb)
- Benchmark comparatif sur une dataset de patch 256x256 (variation du JPEG): [lien benchmark](expe_compression_classique/02_benchmark_compression_dataset.ipynb)
- Benchmark comparatif sur une dataset de patch 2048×2048 (JPEG, WebP, JPEG2K, JPEG XL, AVIF): [lien benchmark](expe_compression_classique/benchmarkJPEG_WEBP_JPEG2000.ipynb)
- Benchmark SVG vectoriel avec VTracer: [lien benchmark](expe_compression_classique/test_format_svg.ipynb)
- Reproduction du papier CINR (Lee et al., MICCAI 2024) sur Apple Silicon (MPS). Modèle : Fourier features + SineCNN sur un patch 256×256: [lien vers demo CINR](cinr-repro/notebooks/01_demo_tile.ipynb)

### Données utilisées
- **91 lames SVS TCGA** 
- Données traitées: 
    - Patches d'entraînement : ~30 000 tuiles 256×256 extraites des zones tissulaires
    - Patches d'entraînement : 500 tuiles 2048x2048 extraites des zones tissulaires
    - voir sur mon compte HF: [lien des dataset](https://huggingface.co/datasets/nathbns/)

---
