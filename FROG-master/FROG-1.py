import nibabel as nib
import os
import argparse  # Ajout de la bibliothèque pour les arguments
from os import listdir
from os.path import abspath, join
import time
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Script de transformation inverse de masques et création de cartes de score.")
    
    # Définition des arguments
    parser.add_argument("--FROG_path", required=True, help="Chemin vers le dossier de FROG")
    parser.add_argument("--output_scoreMap", required=True, help="Dossier final pour les cartes de scores")
    parser.add_argument("--images_rg", required=True, help="Dossier des images de référence")
    parser.add_argument("--images_tr", required=True, help="Dossier des images cibles")
    parser.add_argument("--bin", required=True, help="Chemin vers l'exécutable VolumeTransform")
  

    return parser.parse_args()

# --- Fonctions utilitaires (inchangées) ---
def separate():
    print("******************************************************")

def execute(cmd):
    start_time = time.time()
    separate()
    print("Executing : " + cmd)
    code = os.system(cmd)
    if code: raise OSError(f"Command failed with code {code}")
    print(f"Executed in {round(time.time()- start_time)}s")

# --- Logique principale ---
if __name__ == "__main__":
    args = get_args()

   
    dummyFile = join( args.FROG_path, "output", "dummy.mhd")
    outputMasks = args.output_scoreMap
    imagesRgDir = args.images_rg
    labelsRgDir = imagesRgDir.replace('imagesRg', 'labelsRg')
    transformBin = args.bin
    


    os.makedirs(outputMasks, exist_ok=True)

    # Chargement des masques de référence
    reference_mask = []
    transRef = []
    for i, f in enumerate(sorted(listdir(imagesRgDir))):
        for ext in [".nii.gz", ".mhd"]:
            if f.endswith(ext):
                reference_mask.append(abspath(join(labelsRgDir, f)))
                transRef.append(i)

    # Étape 1 : Déplace les masques de référence dans l'espace commun
    for i in range(len(reference_mask)):
        mask_path = reference_mask[i]
        json_path = join(args.FROG_path, "output", "transforms", f"{transRef[i]}.json")
        out_path = join(args.FROG_path, "output", f"transformed_mask{i}.nii.gz")
        
        execute(f"{transformBin} {mask_path} {dummyFile} -t {json_path} -o {out_path} -i nearestneighbour")

    # Étape 2 : Traitement des images cibles
    imagesTrDir = args.images_tr
    imagesTrPaths = []
    imagesTrNames = []

    for f in sorted(listdir(imagesTrDir)):
        if f.endswith(".nii.gz") or f.endswith(".mhd"):
            imagesTrPaths.append(abspath(join(imagesTrDir, f)))
            imagesTrNames.append(f)

    duration = []
    for i in range(len(imagesTrNames)):
        start_time = time.time()
        transformed_mask_target = join(outputMasks, imagesTrNames[i])
        
        #déplace les masques de référence de l'espace commun à l'espace de l'image cible
        for k in range(len(reference_mask)):
            temp_mask_path = join(args.FROG_path, "output", f"transformed_maskTemp{k}.nii.gz")
           
            json_tr_path = join(args.FROG_path, "transformsR", f"{i}.json") 
            
            cmd = f"{transformBin} {join(args.FROG_path, 'output', f'transformed_mask{k}.nii.gz')} {imagesTrPaths[i]} -ti {json_tr_path} -o {temp_mask_path} -i nearestneighbour"
            execute(cmd)

        # Superposition des masques transposés
        masks_data = []
        last_affine = None
        for k in range(len(reference_mask)):
            temp_mask_path = join(args.FROG_path, 'output', f"transformed_maskTemp{k}.nii.gz")
            img = nib.load(temp_mask_path)
            masks_data.append(img.get_fdata().astype(np.uint8))
            last_affine = img.affine

        final_mask_stack = np.stack(masks_data, axis=0)
        final_img = nib.Nifti1Image(final_mask_stack, last_affine)
        nib.save(final_img, transformed_mask_target)
        
        duration.append(round(time.time() - start_time))

    print(f"Moyenne : {np.mean(duration)}s | Std : {np.std(duration)}s")
