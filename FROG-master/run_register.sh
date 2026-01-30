#!/bin/bash


if [ "$#" -ne 3 ]; then
    echo "Usage: $0 </chemin/vers/imagesRg> <chemin/vers/images_tr> <chemin/vers/dossier/output/>"
    exit 1
fi

IMAGES_RG=$1
IMAGES_TR_DIR=$2
OUTPUT_DIR = $3

# Compter le nombre de masques de référence
count=$(ls "$IMAGES_RG" | wc -l)

i=0
#pour toutes les images à segmenter
for f in "$IMAGES_TR_DIR"/*.nii.gz; do

    # Lancer le script Python qui recale les images cibles sur l'espace commun. Le dossier output est le dossier dans lequel on travail
    python3 tools/register.py -d OUTPUT_DIR "$f" -cmin -100 -cmax 500 --orientation "RAS" 

    # Par défaut register.py met le fichier .json de la transformation dans le dossier transforms et la nomme en fonction du nombre de transformation déjà calculé lors du premier recalage
    # Si on a 15 masques de référence, les transformations vont toutes s'écraser dans le dossier transforms sous le nom de 15.json
    #On déplace donc les fichiers de transformation dans le dossier transformsR et on les numérote de 0 à len(imagesTr) -1

    mv ./transforms/$count.json ./transformsR/$i.json
   
    i=$((i + 1))
done

