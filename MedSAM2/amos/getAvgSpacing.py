import os
import nibabel as nib
import numpy as np

def get_spacing_from_nii(file_path):
    img = nib.load(file_path)
    return img.header.get_zooms()[:3]  # x, y, z spacing

def average_spacing_in_folder(folder_path):
    spacings = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            full_path = os.path.join(folder_path, filename)
            try:
                spacing = get_spacing_from_nii(full_path)
                spacings.append(spacing)
                print(f"{filename} => Spacing: {spacing}")
            except Exception as e:
                print(f"Erreur lors de la lecture de {filename} : {e}")

    if spacings:
        
        spacings_array = np.array(spacings)

        mean_spacing = np.mean(spacings_array, axis=0)
        max_spacing = np.max(spacings_array, axis=0)
        min_spacing = np.min(spacings_array, axis=0)

        return tuple(mean_spacing), tuple(max_spacing), tuple(min_spacing)
       
    else:
        return None

# Exemple d'utilisation
dossier = "/home/friot/MedSAM2/MedSAM2/amostest/imagesTr"
spacing_moyen, max_spacing, min_spacing = average_spacing_in_folder(dossier)

if spacing_moyen:
    print("\n📊 Résumé des spacings :")
    print(f"🔹 Spacing moyen (x, y, z) : {spacing_moyen}")
    print(f"🔺 Spacing max   (x, y, z) : {max_spacing}")
    print(f"🔻 Spacing min   (x, y, z) : {min_spacing}")
else:
    print("Aucune image .nii.gz valide trouvée.")