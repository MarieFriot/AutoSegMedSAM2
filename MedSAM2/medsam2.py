

#shrink les valeure minimalee et maximalee d'intensité
#Normalisation entre 0 et 255
#Codage de l'image en entier (np.uint8)

#récupérer la slice du milieu key_slice_img 

#récupérer les coordonées de la bbox (nombre entier)  x_mi, y_min, x_max, y_max bbox = np.array([bbox[1], bbox[0], bbox[3], bbox[2]])

#video_height = key_slice_img.shape[0] ---> ICI 1
#video_width = key_slice_img.shape[1] --> 512 (largeur de l'image 2D)

#convertir l'image originale en RGB et et la resize de taille 512x512
# normaliser l'image entre 0 et 1 (divise par 255)
#convertir l'image en torch et la mettre sur le gpu
#mean = (0.485, 0.456, 0.406).485 et std = (0.229, 0.224, 0.225)
# z-score normalization

## Mode inférence et autocast
#initier le statut : inference_state = predictor.init_state(img_resized, video_height, video_width)

# inference


#si on a un masque de segmentation :
    #extrait la plus grande composante connexe
    #transforme en entier
import gc
import SimpleITK as sitk
import os
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2_video_predictor_npz
from skimage import measure
import sys
import torch.nn.functional as F
import logging
from codecarbon import OfflineEmissionsTracker
from torchinfo import summary
from scipy.special import softmax

def mask3D_to_bbox(gt3D, max_shift=20):
    """
    Input : masque de segmentation 3D binaire
    Calcul une boîte englobante 3D à partir d'un masque de segmentation binaire
    Output : coordonnées d'une boîte englobante 3D [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    if (len(z_indices)) == 0 :
        return np.array([])
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    D, H, W = gt3D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d

def mask3D_to_bbox_extended(gt3D, increase_ratio=0.2):
    """
    Input : masque de segmentation 3D binaire
    Augmente de increase_ration% la hauteur zmax-zmin
    Output : coordonnées d'une boîte englobante 3D [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    if len(z_indices) == 0:
        return np.array([])

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)

    D, H, W = gt3D.shape

    # Étendre la hauteur (profondeur) sur l'axe Z
    depth = z_max - z_min + 1
    new_depth = int(round(depth * (1 + increase_ratio)))
    cz = (z_min + z_max) / 2

    z_min_new = int(round(cz - new_depth / 2))
    z_max_new = int(round(cz + new_depth / 2))

    # Clamp dans les limites de l'image
    z_min_new = max(0, z_min_new)
    z_max_new = min(D - 1, z_max_new)

    boxes3d = np.array([
        x_min, y_min, z_min_new,
        x_max, y_max, z_max_new
    ])
    return boxes3d

def mask2D_to_bbox(gt2D, max_shift=20):
    """
    Input : segmentation 2D binaire
    Défini une boîte englobante à partir d'une segmentation binaire
    Output : coordonée de la boîte englobante sous la forme [x_min, y_min, x_max, y_max]
    """
    y_indices, x_indices = np.where(gt2D > 0)
    if (len(y_indices)) == 0 :
        return np.array([])
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    bbox_shift = max_shift
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask2D_to_bbox_shifted(gt2D, increase_ratio=0.1):
    """
    Input : segmentation 2D binaire
    Calcul et augmente de increase_ration% l'aire de la boîte englobante 2D
    Output : coordonée de la boîte englobante sous la forme [x_min, y_min, x_max, y_max]
    """
    y_indices, x_indices = np.where(gt2D > 0)
    if len(y_indices) == 0:
        return np.array([])

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = width * height

    new_area = area * (1 + increase_ratio)
    aspect_ratio = width / height

    # Calculer la nouvelle largeur et hauteur
    new_height = np.sqrt(new_area / aspect_ratio)
    new_width = new_height * aspect_ratio

    # Centrer autour du centre initial
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    x_min_new = int(round(cx - new_width / 2))
    x_max_new = int(round(cx + new_width / 2))
    y_min_new = int(round(cy - new_height / 2))
    y_max_new = int(round(cy + new_height / 2))
    #print(((x_max_new - x_min_new) * (y_max_new - y_min_new)) / ((x_max - x_min) * (y_max - y_min)))

    # Clip aux dimensions de l'image
    H, W = gt2D.shape
    x_min_new = max(0, x_min_new)
    y_min_new = max(0, y_min_new)
    x_max_new = min(W - 1, x_max_new)
    y_max_new = min(H - 1, y_max_new)

    boxes = np.array([x_min_new, y_min_new, x_max_new, y_max_new])
    return boxes

def getLargestCC(segmentation):
    """
    Récupère la plus large composante connexe
    """
    # Trouver les composantes connexes dans ce masque binaire
    labels = measure.label(segmentation)
    # Trouver la plus grande composante connexe
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def getLargestCC_multiclass(segmentation):
    """
    Récupère la plus large composante connexe pour chaque classes
    """
    # Créer un masque de sortie vide
    output = np.zeros_like(segmentation)

    # Identifier toutes les classes présentes sauf l'arrière-plan (0)
    classes = np.unique(segmentation)
    classes = classes[classes != 0]

    for cls in classes:
        # Isoler la classe courante
        binary_mask = segmentation == cls

        # Trouver les composantes connexes dans ce masque binaire
        labels = measure.label(binary_mask)

        # Trouver la plus grande composante (ignorer le label 0)
        if labels.max() == 0:
            continue  # Aucun composant trouvé

        largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
        
        # Ajouter cette composante au masque de sortie
        output[largestCC] = cls
        

    return output

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array


def dice_multi_class(preds, targets, class_num, class_count):
    """
    Calcul le score de dices pour chaques classes

    Parameters : 
        preds (np.ndarray avec dtype=uint8) (d,h,w) : Masques de segmentation prédits encodés das un tenseur de 
            taille (d,h,w) avec 0 pour le fond et des nombres entiers pour les autres classes
        targets : format identique à preds. Vérité terrain.
        classs_num (int) : Nombre de classe au total (sans compter le background)
        class_count (np.ndarray) de taille class_num : recense le nombre d'occurence de chaque organes dans les différentes
            images afin de calculer ensuite le dice moyen et l'écart type (sans normaliser par le nombr d'image mais par le nombre
            de fois où un organe apparait dans le dataset)
    """
    smooth = 1e-5
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = np.zeros(class_num)
    for label in labels:
        
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices[label -1] = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        class_count[label -1] +=1
    return np.mean(dices), dices, class_count

#Logger pour les scores de Dice
logger = logging.getLogger("mon_logger")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("dice.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


#Dossier initial
data_root = "/home/friot/AutoSegMedSAM2/MedSAM2/amos"

class_num = 15
#récupérer la liste des images
scan_list = os.listdir(os.path.join(data_root, "imagesTr"))
scan_list.sort()



gt_files =  []    
img_files = []
prompt_files = []

#récupérer les fichiers images et les masques de vérité
#les fichiers de gt et les images doivent porter les memes noms au début 
for scan in scan_list :
    path_tr = os.path.join(data_root, "imagesTr", scan)
    img_files.append(path_tr)
    
    gt_files.append(os.path.join(data_root, "labelsTr", scan))
    #prompt_files.append(os.path.join(data_root, "spacing075-SURF/maskPred3Prompt", scan))
    prompt_files.append(os.path.join(data_root, "scoreMap", scan))

model_cfg = "configs/sam2.1_hiera_t512.yaml"
checkpoint = "checkpoints/MedSAM2_latest.pt"
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)   


#Initialisé CodeCarbon
base_dir = os.path.dirname(os.path.abspath(__file__))  
output_dir = os.path.join(base_dir, "emissions")  
os.makedirs(output_dir, exist_ok=True)
tracker = OfflineEmissionsTracker(output_dir=output_dir, project_name="MedSAM2Inference",tracking_mode="process",save_to_api=False,experiment_id=f"Nombre d'images  : {len(img_files)}", output_file="emissions.csv", country_iso_code="FRA", log_level="warning")
tracker.start()


dice_score = []
class_count = np.zeros(class_num)
max_dice = 0
min_dice = 1

#Seuil pour la carte des scores par organes
threshold = [0.6,0.6,0.4,0.3,0.2,0.2,0.6,0.3,0.3,0.4,0.1,0.1,0.3,0.7,0.1]

#Augmentation de la hauteur en z et des aires des boîtes englobantes

zcoef = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0,0,0.2,0,0,0.2,0.2,0.2]
xycoef = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
#INFERENCE
try : 
    for i in range(0,len(img_files)) :
    
        nii_image = sitk.ReadImage(img_files[i])
        nii_image_data = sitk.GetArrayFromImage(nii_image)

        gt3D = sitk.ReadImage(gt_files[i])
        gt3D = sitk.GetArrayFromImage(gt3D)

        prmpt = sitk.ReadImage(prompt_files[i])
        prmpt = sitk.GetArrayFromImage(prmpt)

        classes = np.unique(gt3D)
        classes = classes[classes != 0 ]

        segs_3D = np.zeros(nii_image_data.shape, dtype=np.uint8)
        segs_logits = torch.zeros((class_num+1, *nii_image_data.shape), dtype=torch.float32).cuda()  # +1 for background

        nii_image_data= np.clip(nii_image_data, -100, 500) #clip l'intensité entre -100 et 500
        nii_image_data_pre = (nii_image_data - np.min(nii_image_data))/(np.max(nii_image_data)-np.min(nii_image_data))*255.0 #Normalisation entre 0 et 2555
        nii_image_data_pre = np.uint8(nii_image_data_pre)
       
    
        video_height = nii_image_data_pre.shape[1]
        video_width = nii_image_data_pre.shape[2]

        img_resized = resize_grayscale_to_rgb_and_resize(nii_image_data_pre, 512)
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        img_mean=(0.485, 0.456, 0.406)
        img_std=(0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
        img_resized -= img_mean
        img_resized /= img_std
       
        with torch.inference_mode(),  torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img_resized, video_height, video_width)
          
            for class_id in classes :
              
                class_mask = (prmpt == class_id).astype(np.uint8)
        
                th = threshold[class_id -1]

                if len(class_mask.shape)  == 4 : #si c'est 15 cartes de scores supperposées
                    class_mask = np.mean(class_mask, axis=3) # on les moyenne
                    prob_map_with_bbox = class_mask.copy()
                    class_mask = (class_mask > th).astype(np.uint8)


                
                
                bbox3D = mask3D_to_bbox(class_mask, 0)
                if bbox3D.shape[0] == 0:
                    print(f"OH we lost the organ {class_id} in the mask {prompt_files[i]}")
                    continue

                
                percentages = [0.25,0.5,0.75]
                slices = [int((bbox3D[-1] - bbox3D[2]) * p) + bbox3D[2] for p in percentages]
                bboxes = []
               
                fallback_failed = False 
                for s in slices:
                    bbox = mask2D_to_bbox((prob_map_with_bbox[s, :, :] > th).astype(np.uint8),0)
                    if bbox.shape[0] == 0:
                        print(f"OH weird organ {class_id} in the mask {prompt_files[i]} at slice {s}")
                        bbox = mask2D_to_bbox((prob_map_with_bbox[s, :, :] > 0).astype(np.uint8),0)
                        if bbox.shape[0] == 0:
                            print(f"OH weird organ {class_id} in the mask {prompt_files[i]} at slice {s}")
                            bbox = mask2D_to_bbox((prob_map_with_bbox[slices[1], :, :] > th).astype(np.uint8), 0) #FALLBACK TO MIDDLE BBX
                            if bbox.shape[0] == 0 :
                                fallback_failed = True
                                break
                    bboxes.append((s, bbox))

                if fallback_failed:
                    continue

                # Dessiner les bords pour chaque bbox slice
                for s, bbox in bboxes:
                    prob_map_with_bbox[s, bbox[1]:bbox[3]+1, bbox[0]] = 1.0  # Bord gauche
                    prob_map_with_bbox[s, bbox[1]:bbox[3]+1, bbox[2]] = 1.0  # Bord droit
                    prob_map_with_bbox[s, bbox[1], bbox[0]:bbox[2]+1] = 1.0  # Bord haut
                    prob_map_with_bbox[s, bbox[3], bbox[0]:bbox[2]+1] = 1.0  # Bord bas

                y = np.arange(bbox3D[0], bbox3D[3]+1, 2)
                prob_map_with_bbox[bbox3D[-1],  (bbox3D[4] - bbox3D[1])//2 + bbox3D[1], y] = 1.0 #Zmax
                prob_map_with_bbox[bbox3D[2],  (bbox3D[4] - bbox3D[1])//2 + bbox3D[1], y] = 1.0 #Zmin

                # Sauvegarde de l'image avec les bordures de bbox
                sitk_prob_with_bbox = sitk.GetImageFromArray(prob_map_with_bbox)
                sitk_prob_with_bbox.CopyInformation(nii_image)
                mask_path = gt_files[i]
                output_path = mask_path.replace(
                    'labelsTr',
                    f'probMap/{class_id}'
                )
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                sitk.WriteImage(sitk_prob_with_bbox, output_path)


                #----------------------------- INFERENCE AVEC 3 BOITES ENGLOBANTES-----------------------------------------------
                #partie haut
                max_frame_num_to_track = (bbox3D[-1] - bboxes[1][0] ) # zmax - zmilieu
                _, out_obj_ids, out_mask_logits  = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=bboxes[1][0], obj_id=1, box=bboxes[1][1])
                _, out_obj_ids, out_mask_logits  = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=bboxes[2][0], obj_id=1, box=bboxes[2][1])
            
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, max_frame_num_to_track = max_frame_num_to_track):
                    segs_logits[int(class_id), out_frame_idx ] += out_mask_logits[0][0].float()
                    
                predictor.reset_state(inference_state)
             
                #partie bas
                max_frame_num_to_track = (bboxes[1][0] - bboxes[0][0])# zmilieu à z0.25
                
                _, out_obj_ids, out_mask_logits  = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=bboxes[1][0], obj_id=1, box=bboxes[1][1])
               
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse = True, max_frame_num_to_track = max_frame_num_to_track):
                    segs_logits[int(class_id), out_frame_idx ] += out_mask_logits[0][0].float()


                max_frame_num_to_track = (bboxes[1][0] -bbox3D[2])# zmilieu à zmin
                
                _, out_obj_ids, out_mask_logits  = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=bboxes[0][0], obj_id=1, box=bboxes[0][1])
               
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse = True, max_frame_num_to_track = max_frame_num_to_track):
                    segs_logits[int(class_id), out_frame_idx ] += out_mask_logits[0][0].float()
                    if out_frame_idx < bboxes[0][0] : 
                        segs_logits[int(class_id), out_frame_idx ] += out_mask_logits[0][0].float()

                predictor.reset_state(inference_state)

                """-------------------------INFERENCE 1 BOITE ENGLOBANTE-----------------------------------------------------
                #partie haut
                max_frame_num_to_track = (bbox3D[-1] - bboxes[1][0] ) # zmax - zmilieu
                _, out_obj_ids, out_mask_logits  = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=bboxes[1][0], obj_id=1, box=bboxes[1][1])
            
            
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, max_frame_num_to_track = max_frame_num_to_track):
                    segs_logits[int(class_id), out_frame_idx ] += out_mask_logits[0][0].float()
                    
                predictor.reset_state(inference_state)
             
                #partie bas
                max_frame_num_to_track = (bboxes[1][0] - bbox3D[2])# zmilieu à zmin
                
                _, out_obj_ids, out_mask_logits  = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=bboxes[1][0], obj_id=1, box=bboxes[1][1])
               
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse = True, max_frame_num_to_track = max_frame_num_to_track):
                    segs_logits[int(class_id), out_frame_idx ] += out_mask_logits[0][0].float()
             
                predictor.reset_state(inference_state)
                """"
             
        segs_logits = segs_logits.to('cpu')     
        segs_prob = F.softmax(segs_logits, dim=0)
        segs_3D = torch.argmax(segs_prob, dim=0).cpu().numpy().astype(np.uint8)
    
        if np.max(segs_3D) > 0:
            segs_3D = getLargestCC_multiclass(segs_3D)
            segs_3D = np.uint8(segs_3D)

    
        mean_dice, dices, class_count = dice_multi_class(segs_3D, gt3D, class_num, class_count)
        if mean_dice > max_dice : 
            print(f"------------> Nouveau meilleur cas {gt_files[i]} :)")
            max_dice = mean_dice
        elif mean_dice < min_dice :
            print(f"------------> Oh non un nouveau pire cas {gt_files[i]} :(")
            min_dice = mean_dice
        
        if len(dice_score) == 0 :
            dice_score = [dices]
        else :
            dice_score.append(dices)
  
        sitk_mask = sitk.GetImageFromArray(segs_3D)
        sitk_mask.CopyInformation(nii_image)
        mask_path = gt_files[i]
        sitk.WriteImage(sitk_mask, mask_path.replace('labelsTr', f"maskPred") ) #Remplacer par le dossier dans lequel sera envoyé les segmentations

        print(f"{prompt_files[i]} : {np.round(dices,2)}")
        logger.debug(f"Dice score: {np.round(dices, 2)}")
        logger.debug(f"Class_count : {class_count} ")

        del segs_logits, img_resized, img_mean, img_std, segs_prob, prmpt, prob_map_with_bbox
        gc.collect()
        torch.cuda.empty_cache()
 

    dice_score = np.stack(dice_score)
    dice_score = np.ma.masked_equal(dice_score, 0)
    meanDice = np.round(dice_score.sum(axis=0)/class_count, 2)
    logger.debug(f"Mean dice score {th}: {meanDice}, {np.mean(meanDice)}")
    ecart_type = np.ma.std(dice_score, axis=0).filled(np.nan)
    logger.debug(f"Variance dice score {th}: {np.round(ecart_type,2)}")
finally : 

    tracker.stop()
