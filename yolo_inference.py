import numpy as np
from PIL import Image
import nibabel as nib
from ultralytics import YOLO
import os
import json
from codecarbon import OfflineEmissionsTracker
import torch
import time


def convertToLAS(filename):
    # Charger l'objet image
    imageObj = nib.load(filename)

    # Récupérer l'orientation actuelle
    current_orientation = nib.aff2axcodes(imageObj.affine)
    print(f"Orientation actuelle de {filename} : {current_orientation}")

    # Vérifier si l'image est déjà en LAS
    if current_orientation == ('L', 'A', 'S'):
        print("L'image est déjà en orientation LAS, aucune action nécessaire.")
        return filename

    # Définir l'orientation cible (LAS)
    target_orientation = ('L', 'A', 'S')

    # Créer la matrice de transformation vers LAS
    # On calcule la transformation nécessaire entre l'affine actuel et le LAS
    ornt_current = nib.io_orientation(imageObj.affine)
    ornt_target = nib.orientations.axcodes2ornt(target_orientation)
    transform = nib.orientations.ornt_transform(ornt_current, ornt_target)

    # Appliquer la transformation (réorganisation des données et de l'affine)
    flippedImage = imageObj.as_reoriented(transform)

    # Préparer la nouvelle image Nifti
    img_data = flippedImage.get_fdata()
    img_conv = nib.Nifti1Image(
        img_data.astype(flippedImage.header.get_data_dtype()),
        flippedImage.affine,
        flippedImage.header
    )

    # S'assurer que les codes qform/sform sont corrects pour la lecture par les logiciels tiers
    img_conv.header['qform_code'] = 1
    img_conv.header['sform_code'] = 1


    new_orientation = nib.aff2axcodes(img_conv.affine)
    print(f"Nouvelle orientation : {new_orientation}")

    return img_conv


MODEL_PATH = './bestYOLOV11n_pretrained.pt'
OUTPUT_JSON_FILE = './3d_bounding_boxes_results_v11n3P.json'
data_root = "/home/marfriot/prjrech/MedSAM2/amos"
output_dir = "./codecarbon"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Grande liste pour stocker TOUTES les prédictions 2D de TOUTES les tranches
all_predictions_2d = []
# Nouvelle liste pour stocker les résultats JSON de chaque image
all_images_results = []

scan_list = os.listdir(os.path.join(data_root, "imagesVa"))
image_files = []

for scan in scan_list :

  path = os.path.join(data_root, "imagesVa", scan)
  image_files.append(path)

#TARGET_RATIOS = [0.17, 0.34, 0.5, 0.66, 0.83]
#RATIO_NAMES = ["17pc", "34pc", "50pc", "66pc", "83pc"]
TARGET_RATIOS = [0.25, 0.5, 0.75]
RATIO_NAMES = ["25pc","50pc", "75pc"]

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# CT windowing
WINDOW_CENTER = 40
WINDOW_WIDTH = 400

# Class names
CLASS_NAMES = {
    0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney",
    4: "gall bladder", 5: "esophagus", 6: "liver", 7: "stomach",
    8: "arota", 9: "postcava", 10: "pancreas", 11: "right adrenal gland",
    12: "left adrenal gland", 13: "duodenum", 14: "bladder", 15: "prostate/uterus"
}

# ==================== HELPER FUNCTIONS ====================

def normalize_slice(slice_2d, window_center=40, window_width=400):
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    slice_2d = np.clip(slice_2d, min_value, max_value)
    slice_2d = (slice_2d - min_value) / (max_value - min_value) * 255
    return slice_2d.astype(np.uint8)

# ==================== LOAD MODEL ====================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH).to(device)
print("âœ“ Model loaded")
duration = []
tracker = OfflineEmissionsTracker(output_dir=output_dir, project_name="YOLOv11n",tracking_mode="process",save_to_api=False,experiment_id=f"Nombre d'images  : {len(image_files)}", output_file="emissions.csv", country_iso_code="FRA", log_level="warning")
tracker.start()
for f in range(len(image_files)) :
  all_predictions_2d = []

  # ==================== LOAD NIFTI ====================
  NIFTI_PATH = image_files[f]
  print("Loading NIfTI file...")
  #nii = nib.load(NIFTI_PATH)
  nii = convertToLAS(NIFTI_PATH)
  print("âœ“ NIfTI file loaded")
  data = nii.get_fdata()

  num_slices_z = data.shape[2]
  for z in range(num_slices_z):

    # ==================== EXTRACT SLICE ====================

    slice_2d = data[:, :, z]
    normalized_img = normalize_slice(slice_2d, WINDOW_CENTER, WINDOW_WIDTH)

    img_pil = Image.fromarray(normalized_img, mode='L')
    W = img_pil.size[0]

    # ==================== YOLO INFERENCE ====================
    start_time = time.time()
    results = model.predict(
        img_pil,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
        device=device
    )

     # --- EXTRACTION ET STOCKAGE 2D ---
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            # La bbox est stockée comme un numpy array ici
            all_predictions_2d.append({
                    'z_index': z,
                    'bbox_xyxy': box.xyxy[0].cpu().numpy(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0])
                })

  print(f"✓ All {num_slices_z} slices processed for {NIFTI_PATH}!\n")


  organ_data = {}
  for pred in all_predictions_2d:
    class_id = pred['class_id']
    z = pred['z_index']
    bbox = pred['bbox_xyxy']

    if class_id not in organ_data:
        organ_data[class_id] = {
            'z_min': z,
            'z_max': z,
            'detections': [(z, bbox)]
        }
    else:
        if z < organ_data[class_id]['z_min']:
            organ_data[class_id]['z_min'] = z
        if z > organ_data[class_id]['z_max']:
            organ_data[class_id]['z_max'] = z
        organ_data[class_id]['detections'].append((z, bbox))


    # ==================== CONSOLIDATION 3D (Étape 2 : Calcul et Formatage JSON) ====================


  # ... (votre code précédent)

  current_image_results = {
      'image_name': NIFTI_PATH,
      'organs': {}  # On change la liste [] en dictionnaire {}
  }

  for class_id, data in organ_data.items():
      bboxes = []
      class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")

      # Calcul de la médiane (votre logique actuelle)
      z_min, z_max = data['z_min'], data['z_max']
      height = z_max - z_min

      for ratio, name in zip(TARGET_RATIOS, RATIO_NAMES):
        # Calcul de la cible théorique en Z
        z_target = z_min + (height * ratio)

        closest_bbox = None
        min_diff_z = float('inf')
        actual_z = 0

        # Recherche de la détection existante la plus proche de la cible
        for z_det, bbox_det in data['detections']:
            diff = abs(z_det - z_target)
            if diff < min_diff_z:
                min_diff_z = diff
                closest_bbox = bbox_det
                actual_z = z_det

        # Transformation des coordonnées LAS vers RAS (si nécessaire comme dans votre code)
        if closest_bbox is not None:
            y1, x1_las, y2, x2_las = closest_bbox
            x1_ras = W - x2_las
            x2_ras = W - x1_las
            formatted_bbox = [int(x1_ras), int(y1), int(x2_ras), int(y2)]
        else:
            formatted_bbox = [0, 0, 0, 0]

        bboxes.append((actual_z, formatted_bbox))


      # ON UTILISE L'ID COMME CLÉ (converti en string pour le format JSON)
      current_image_results['organs'][str(class_id)] = {
          'class_name': class_name,
          'z_min': int(z_min),
          'z_max': int(z_max),
          'bboxes' : bboxes

      }


  duration.append(time.time()-start_time)
  # Ajouter les résultats de cette image à la liste globale
  all_images_results.append(current_image_results)

  # ==================== ENREGISTREMENT JSON FINAL ====================

  print(f"\n--- WRITING RESULTS TO {OUTPUT_JSON_FILE} ---")

  try:
      with open(OUTPUT_JSON_FILE, 'w') as f:
          # indent=4 rend le fichier lisible
          json.dump(all_images_results, f, indent=4)
      print(f"✓ Successfully wrote {len(all_images_results)} image results to {OUTPUT_JSON_FILE}")
  except Exception as e:
      print(f"!!! Error writing JSON file: {e}")

tracker.stop()

print(duration)
print(np.mean(duration))
print(np.std(duration))
