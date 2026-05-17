## Liens article / site
1. https://www.johnsnowlabs.com/what-to-know-before-de-identifying-whole-slide-images-wsi/
2. https://dicom.nema.org/dicom/dicomwsi/
3. https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_9.html
4. https://arxiv.org/pdf/2503.18074 (meilleur ressource pour l'instant)
5. https://arxiv.org/html/2503.23862v1
6. https://pmc.ncbi.nlm.nih.gov/articles/PMC8525863/

## Liens gh
1. https://github.com/John-P/wsic
2. https://github.com/smujiang/WSI2DICOM 
3. https://github.com/debarron/svs-image-analysis 
4. https://github.com/visioncortex/vtracer

## format des fichier donnée: svs
https://openslide.org/formats/aperio/


### Etat de l'art 

#### 1 Compression generique

- **JPEG 2000** reste le standard de facto en pathologie (Aperio, Philips). Deux profils : irreversible (9/7) et reversible (5/3).
- **JPEG XL** (ISO/IEC 18181) promet des gains significatifs : meilleure qualite subjective, support sans perte, progressive. Peu de validateurs medicaux a ce jour.
- **HEVC-MSP** (Main Still Picture) : etudie pour remplacer JPEG, mais problemes de licences.

#### 2 Compression pour WSI

- **WSI-specific tiling** : les images sont decoupees en tuiles (256x256 ou 512x512). Chaque tuile est compressee independamment. Cela permet l'acces aleatoire rapide.
- **High Throughput JPEG 2000 (HTJ2K)** : version acceleree de J2K (ISO/IEC 15444-15), jusqu'a 10x plus rapide en decodage. Implemente dans `OpenJPH`.
