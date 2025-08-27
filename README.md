# AutoSegMedSAM2

Contenu scientifique décrit dans le fichier rapport_de_stage.pdf.

D'un point de vue pratique, pour réaliser la méthode de segmentation automatique décrite dans le fichier rapport_de_stage.pdf voici les étapes à suivre.


## Arborescence des fichiers
```text
AutoSegMedSAM2/
├── README.md
├── MedSAM2/
│   ├── amos/
│   │   ├── imagesTr/
│   │   └── labelsTr/
│   │   └── imagesRg/
│   │   └── labelsRg/
│   │   └── scoreMap/
│   │
│   └── medsam2.py
├── FROG-MASTER/
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
python3 FROG.py ..../MedSAM2/amos/imagesRg - o output -ras -cmin 100 -cmax 500 -a 5 -lancho
```
## 2- Recalage des images cibles (à segmenter) sur cette espace commun (calcul des transformations associées)
```text
./run_register.sh
```
## 3- Recalage des masques de références sur les images cibles : création des cartes de score
```text
python3 FROG-1.py
```
## 4 - Segmentations avec MedSAM2
```text
python3 medsam2.py
```
