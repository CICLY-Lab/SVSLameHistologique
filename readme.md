# Compression d'Images SVS/DICOM

### Rapport détaillé du SOTA + source utiles sur les liens ci-dessous:
[liens vers SOTA](rapport_etat_art.md) <br/>
[liens vers source utile](source_utiles.md)

## Contexte
Les Whole Slide Images (WSI) utilisées en anatomopathologie numérique atteignent 
des résolutions extrêmes, générant des fichiers 
de plusieurs gigaoctets. 

## Objectifs du stage
1. Analyser les formats et propriétés des images WSI (svs/dicom)
2. Benchmarker les codecs de compression classiques (JPEG, JPEG2000, JPEG XL, etc.)
3. Concevoir une méthode de compression adaptative aux caractéristiques des lames
4. Évaluer l'impact qualitatif des compressions sur le diagnostic
5. Implémenter un modèle inspiré de CLERIC / CINR ?

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

**Semaine 5** (en cours):
- Analyser les fichier DICOM du labo (nv de compression, etc..) (FAIT)
- MRXS (comparer avec le svs/dicom)
- CINR sur un plus grand fichier (EN COURS)
- Tester les nouvelles piste de svg


### Données utilisées
- **91 lames SVS TCGA** 
- Données traitées: 
    - Patches d'entraînement : ~30 000 tuiles 256×256 extraites des zones tissulaires
    - voir sur mon compte HF: [lien du dataset](https://huggingface.co/datasets/nathbns/SVS-TCGA-BR)

---
