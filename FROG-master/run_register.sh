#!/bin/bash


# Compter le nombre de masques de référence
count=$(ls ./../MedSAM2/amos/imagesRg/ | wc -l)

i=0
#pour toutes les images à segmenter
for f in ./../MedSAM2/amos/imagesTr/*.nii.gz; do

    # Lancer le script Python qui recale les images cibles sur l'espace commun. Le dossier output est le dossier dans lequel on travail
    python3 tools/register.py -d ./output -i "$f" -cmin -100 -cmax 500 --orientation "RAS" 

    # Par défaut register.py met le fichier .json de la transformation dans le dossier transforms et la nomme en fonction du nombre de transformation déjà calculé lors du premier recalage
    # Si on a 15 masques de référence, les transformations vont toutes s'écraser dans le dossier transforms sous le nom de 15.json
    #On déplace donc les fichiers de transformation dans le dossier transformsR et on les numérote de 0 à len(imagesTr) -1

    mv ./transforms/$count.json ./transformsR/$i.json
   
    i=$((i + 1))
done

