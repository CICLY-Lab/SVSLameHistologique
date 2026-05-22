# Guide pratique : utiliser le cluster P2CHPD

> Document pour vous connecter au cluster, installer votre environnement Python, et lancer un entraînement GPU.

---

## Sommaire

1. [Avant de commencer](#1-avant-de-commencer)
2. [Se connecter au cluster](#2-se-connecter-au-cluster)
3. [Comprendre le cluster](#3-comprendre-le-cluster)
4. [Transférer des fichiers](#4-transférer-des-fichiers)
5. [Installer son environnement Python](#5-installer-son-environnement-python)
6. [Créer un script de job](#6-créer-un-script-de-job)
7. [Soumettre et suivre son job](#7-soumettre-et-suivre-son-job)
8. [Récupérer ses résultats](#8-récupérer-ses-résultats)
9. [Aide-mémoire des commandes](#10-aide-mémoire-des-commandes)

---

## 1. Avant de commencer

### Votre identifiant

Votre login est au format `prenom.nom` (celui de l'université).

---

## 2. Se connecter au cluster

### Depuis Mac/Linux

```bash
ssh prenom.nom@p2chpd-login1.univ-lyon1.fr
```

La première fois, le serveur vous demandera de vérifier son empreinte. Tapez `yes`.

> **serveur**
> - `p2chpd-login1.univ-lyon1.fr` / AlmaLinux 8

---

## 3. Comprendre le cluster

Le cluster P2CHPD est composé de plusieurs types de machines :

| Type | Rôle | Ce qu'on peut y faire |
|------|------|----------------------|
| **Nœud de login** | Porte d'entrée | Se connecter, transférer des fichiers, écrire des scripts, soumettre des jobs |
| **Nœud de calcul** | Machines de travail | Exécuter vos programmes (Python, CUDA, etc.) |

### Règles 

- Votre espace personnel (`/home_nfs`) est **sauvegardé** mais **lent** et **interdit d'exécution** (pas de `pip`, pas de `python` depuis cet espace).

### Où stocker ses fichiers

| Emplacement | Capacité | Sauvegardé | Exécutable | Remarque |
|-------------|----------|------------|------------|----------|
| `/home_nfs/groupe/prenom.nom/` | 50 Go (quota) | Oui | **Non (noexec)** | Votre home, pour le code et les résultats |
| `/tmp` | >100 Go (local au nœud) | Non | Oui | Espace de travail rapide, effacé à la fin du job |
| `/scratch` | 40 To (partagé) | Non | Oui | Pas disponible sur tous les nœuds |

> **Le piège principal** : `/home_nfs` est monté en `noexec`. Cela signifie que vous ne pouvez pas exécuter de programmes (Python, pip, conda) depuis votre home. Il faut installer dans `/tmp`.

---

## 4. Transférer des fichiers

### Depuis votre ordinateur vers le cluster

```bash
# Un fichier
scp mon_fichier.py prenom.nom@p2chpd-login1.univ-lyon1.fr:~/

# Un dossier entier
scp -r mon_dossier/ prenom.nom@p2chpd-login1.univ-lyon1.fr:~/
```

### Du cluster vers votre ordinateur

```bash
scp prenom.nom@p2chpd-login1.univ-lyon1.fr:~/resultats/model.pth ./
```

### rsync (recommandé pour les gros dossiers)

```bash
rsync -avz mon_dossier/ prenom.nom@p2chpd-login1.univ-lyon1.fr:~/mon_dossier/
```

---

## 5. Installer son environnement Python

Les nœuds de calcul ont Python 3.6 (trop vieux). On utilise les **modules EasyBuild** pour avoir Python 3.13, puis on crée un environnement virtuel dans `/tmp`.

### Étape 5.1 — Réserver un nœud GPU en interactif

```bash
salloc -p r740-gpu -t 2:00:00 --mem=64G -c 8
```

Attendez le message `Nodes r740-gpu-00X are ready for job`, puis connectez-vous dessus :

```bash
ssh r740-gpu-00X    # remplacez 00X par le nœud alloué
```

### Étape 5.2 — Charger les modules et créer l'environnement

```bash
# Charger les modules EasyBuild
module use /easybuild/AlmaLinux/8/x86_64/eth/modules/all/Core
module load foss/2025b
module load Python/3.13.5

# Créer un environnement virtuel dans /tmp (obligatoire, car /home_nfs est noexec)
python3 -m venv /tmp/$USER/.venv

# Activer l'environnement
source /tmp/$USER/.venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip
```

### Étape 5.3 — Installer les paquets Python

```bash
# PyTorch avec support GPU (CUDA 12.4, compatible A100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Autres dépendances
pip install datasets matplotlib numpy openslide-bin openslide-python pillow scikit-image tqdm huggingface_hub # etc
```

### Étape 5.4 — Vérifier que le GPU est détecté

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Vous devriez voir :
```
CUDA: True
GPU: NVIDIA A100-PCIE-40GB # ou autre
```

### Étape 5.5 — Quitter le nœud

```bash
exit    # quitte le nœud de calcul
exit    # libère l'allocation salloc
```

---

## 6. Créer un script de job

Un « job » est un script qui décrit ce que vous voulez exécuter et les ressources nécessaires. On le soumet à SLURM qui le lance automatiquement sur un nœud de calcul.

### Structure d'un script de job

Créez un fichier `mon_job.sbatch` dans votre home :

```bash
cat > ~/mon_job.sbatch << 'EOF'
#!/bin/bash
#SBATCH -J mon_job              # Nom du job
#SBATCH -p r740-gpu             # Partition (r740-gpu = nœuds avec A100)
#SBATCH -t 2:00:00              # Durée max (2h ici)
#SBATCH --mem=64G               # Mémoire RAM
#SBATCH -c 8                    # Nombre de cœurs CPU
#SBATCH -o %j.out               # Fichier de sortie standard
#SBATCH -e %j.err               # Fichier d'erreurs

# Charger les modules
module use /easybuild/AlmaLinux/8/x86_64/eth/modules/all/Core
module load foss/2025b
module load Python/3.13.5

# Dossier temporaire sur /tmp (rapide, exécutable)
export TMPDIR=/tmp/$USER/${SLURM_JOB_ID}
mkdir -p $TMPDIR

# Activer l'environnement Python
export MY_VENV=/tmp/$USER/.venv
if [ ! -f $MY_VENV/bin/activate ]; then
    # Si l'environnement n'existe pas, le créer (sécurité)
    python3 -m venv $MY_VENV
    source $MY_VENV/bin/activate
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install datasets matplotlib numpy openslide-bin openslide-python pillow scikit-image tqdm huggingface_hub
else
    source $MY_VENV/bin/activate
fi

# Aller dans le dossier du projet
cd ~/mon_projet

# === VOTRE CODE ICI ===
python3 train.py --epochs 100000 --device cuda --output-dir outputs/run_${SLURM_JOB_ID}

# Nettoyer les fichiers temporaires
rm -rf $TMPDIR
EOF
```

### Les partitions disponibles

| Partition | Type de nœud | GPU | Durée max | Nombre de nœuds |
|-----------|-------------|-----|-----------|-----------------|
| `debug` | CPU (C6420) | Non | 1 jour | 4 |
| `c6420-ib100` | CPU (C6420, InfiniBand) | Non | 2 jours | ~30 |
| `c6525-10g` | CPU (AMD C6525) | Non | 2 jours | 2 |
| `r740-gpu` | GPU (Dell R740, **A100 40 Go**) | Oui | 1 jour | 2 |
| `dl385-gpu` | GPU (HPE DL385) | Oui | 1 jour | 2 |
| `lenovo` | GPU (Lenovo SR655) | Oui | 128 jours | 1 |

### Les directives SLURM les plus utiles

| Directive | Exemple | Signification |
|-----------|---------|---------------|
| `#SBATCH -J` | `-J mon_job` | Nom du job |
| `#SBATCH -p` | `-p r740-gpu` | Partition cible |
| `#SBATCH -t` | `-t 2:00:00` | Durée max (2h) |
| `#SBATCH --mem` | `--mem=64G` | Mémoire RAM demandée |
| `#SBATCH -c` | `-c 8` | Nombre de cœurs CPU |
| `#SBATCH -o` | `-o %j.out` | Fichier de sortie (`%j` = numéro du job) |
| `#SBATCH -e` | `-e %j.err` | Fichier d'erreurs |
| `#SBATCH --mail-type` | `--mail-type=END,FAIL` | Envoyer un mail à la fin ou en cas d'erreur |
| `#SBATCH --mail-user` | `--mail-user=prenom.nom@univ-lyon1.fr` | Adresse mail |

---

## 7. Soumettre et suivre son job

### Soumettre un job

```bash
sbatch mon_job.sbatch
```

SLURM vous répond : `Submitted batch job 103456` (le numéro est votre identifiant de job).

### Voir l'état de ses jobs

```bash
squeue -u $USER
```

La colonne `ST` (State) indique :
- `PD` = Pending (en attente de ressources)
- `R` = Running (en cours)
- `CG` = Completing (se termine)
- `CD` = Completed (terminé normalement)
- `F` = Failed (erreur)

### Suivre la sortie en direct

```bash
tail -f 1034XX.out    # sortie standard (remplacez par votre numéro de job)
tail -f 1034XX.err    # erreurs
```

Pour arrêter le suivi : `Ctrl+C`.

### Annuler un job

```bash
scancel 103456    # remplacez par votre numéro de job
```

### Voir les infos détaillées d'un job

```bash
scontrol show job 103456
```

---

## 8. Récupérer ses résultats

Les résultats écrits dans votre home (`/home_nfs/...`) sont accessibles depuis le nœud de login. Pour les récupérer sur votre ordinateur :

```bash
# Depuis votre ordinateur
scp prenom.nom@p2chpd-login1.univ-lyon1.fr:~/mon_projet/outputs/run_103456/model.pth ./
scp prenom.nom@p2chpd-login1.univ-lyon1.fr:~/mon_projet/outputs/run_103456/reconstruction.png ./
```

Ou tout un dossier :

```bash
scp -r prenom.nom@p2chpd-login1.univ-lyon1.fr:~/mon_projet/outputs/ ./
```

---

## 9. Aide-mémoire des commandes

### Connexion et transfert

```bash
ssh prenom.nom@p2chpd-login1.univ-lyon1.fr     # se connecter
scp fichier login@host:~/                        # envoyer un fichier
scp login@host:~/fichier ./                      # récupérer un fichier
rsync -avz dossier/ login@host:~/dossier/        # synchroniser un dossier
```

### SLURM

```bash
sbatch mon_job.sbatch          # soumettre un job
squeue -u $USER                # voir ses jobs
scontrol show job JOBID        # détails d'un job
scancel JOBID                  # annuler un job
sinfo                          # voir les nœuds et partitions
sacct -j JOBID                 # historique d'un job terminé
```

### Modules EasyBuild

```bash
module use /easybuild/AlmaLinux/8/x86_64/eth/modules/all/Core    # activer les modules
module avail                                                    # lister les modules
module load Python/3.13.5                                       # charger un module
module list                                                     # voir les modules chargés
module purge                                                    # décharger tous les modules
```

### Session interactive

```bash
salloc -p r740-gpu -t 2:00:00 --mem=64G -c 8    # réserver un nœud GPU
ssh r740-gpu-00X                                  # se connecter au nœud
exit                                              # quitter le nœud
exit                                              # libérer l'allocation
```

---

## Ressources utiles

- Site du P2CHPD : https://p2chpd.univ-lyon1.fr
- Manuel officiel : https://p2chpd.univ-lyon1.fr/manual
- Documentation SLURM : https://slurm.schedmd.com
- Contact : christophe.pera@univ-lyon1.fr (administrateur du centre de calcul)>
