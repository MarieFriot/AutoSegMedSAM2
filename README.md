# AutoSegMedSAM2

Contenu scientifique décrit dans le fichier rapport_de_stage.pdf.

D'un point de vue pratique, pour réaliser la méthode de segmentation automatique décrite dans le fichier rapport_de_stage.pdf voici les étapes à suivre.

Les méthodes FROG et MedSAM2 sont issus des dépôts suivants : 
  - https://github.com/valette/FROG
  - https://github.com/bowang-lab/MedSAM2


## Arborescence des fichiers
```text
AutoSegMedSAM2/
├── README.md
├── MedSAM2/
│   ├── amos/
│   │   ├── imagesTr/
│   │       └── amos_0309.nii.gz
│   │   └── labelsTr/
│   │       └── amos_0309.nii.gz
│   │   └── scoreMap/
│   │       └── amos_0309.nii.gz
│   │   └── imagesRg/
│   │   └── labelsRg/
│   │
│   └── medsam2.py
│
├── FROG-master/
│   ├── output/
│   │   ├── transforms/
│   │   └── dummy.mhd/
│   │   └── transformed_mask{i}.nii.gz
│   │   └── transformed_maskTemp{i}.nii.gz
│   ├── FROG.py
│   ├── FROG-1.py
│   ├── run_register.sh
│   ├── transforms/
│   └── transformsR/
```

## 1- Recalage des images de référence dans un espace commmun (calcul des transformations associées)
```text
python3 FROG.py AutoSegMedSAM2/MedSAM2/amos/imagesRg -o output -ras -cmin 100 -cmax 500 -a 5 -lanchor 0.5 0.5 0
```
Dans le dossier imagesRg doivent se trouver les images de référence.
Un fichier "dummy.mhd" est automatiquement créé. Il correspond à la description de l'espace commun. 


Si erreur 32512 :  export LD_LIBRARY_PATH=/usr/lib/jvm/java-21-openjdk/lib/server:$LD_LIBRARY_PATH
## 2- Recalage des images cibles (à segmenter) sur cette espace commun (calcul des transformations associées)
```text
./run_register.sh
```
Le script run_register.sh exécute le fichier register.py qui par défaut met les fichiers .json correspondant aux transformations dans le dossier FROG-MASTER/transforms/. Dans run_register.sh les json sont ensuite déplacés dans le dossier FROG-MASTER/transformsR/. Ce dossier comporte donc les transformations des images cibles dans l'espace commun.
## 3- Recalage des masques de références sur les images cibles : création des cartes de score
```text
python3 FROG-1.py
```
Les masques de références dans le dossier labelsRg sont d'abord déplacé dans l'espace commun (transformed_mask{i}.nii.gz) à l'aide des transformations calculées à l'étape 1 et situées dans le dossier FROG-MASTER/output/transforms/.

Ensuite, les masques de référence sont déplacées sur l'image cible (transformed_maskTemp{i}.nii.gz) avec les transformations du dossier FROG-MASTER/transformsR/. Ils sont alors supperposés et empilés dans une seule image de dimension 15xDxHxW dans le dossier scoreMap.

## 4 - Segmentations avec MedSAM2
```text
python3 medsam2.py


```
