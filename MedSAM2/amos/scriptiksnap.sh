#!/bin/bash

# Répertoires
IMAGE_DIR="./amos-suite2"
MASK_DIR="/home/friot/Téléchargements/amos22/amos22/labelsTr"

# Extensions attendues
IMAGE_EXT=".nii.gz"
MASK_EXT=".nii.gz"

# Boucle sur les images
for image_path in "$IMAGE_DIR"/*"$IMAGE_EXT"; do
    image_file=$(basename "$image_path")
    base_name="${image_file%%$IMAGE_EXT}"
    
    mask_path="$MASK_DIR/${base_name}${MASK_EXT}"
    
    if [[ -f "$mask_path" ]]; then
        echo "Ouverture de $image_file avec son masque ${base_name}${MASK_EXT}"
        itksnap -g "$image_path" -s "$mask_path"
        
        # Attente de fermeture d'ITK-SNAP avant de passer au suivant
        read -p "Appuyez sur Entrée pour ouvrir l'image suivante..."
    else
        echo "Aucun masque trouvé pour $mask_path"
    fi
done
