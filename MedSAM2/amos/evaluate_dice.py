import SimpleITK as sitk
import os
import numpy as np
import monai
from monai.metrics import compute_surface_dice
import torch
import torch.nn.functional as F

def to_one_hot(mask, class_num) :
    """
    mask: tensor [H, W, D] avec entiers de 0..class_num-1
    return: [1, C, H, W, D] one-hot
    """
    mask = torch.as_tensor(mask, dtype=torch.long)  # [H,W,D]
    mask = mask.unsqueeze(0).unsqueeze(0)           # [1,1,H,W,D]
    one_hot = F.one_hot(mask, num_classes=class_num)  # [1,1,H,W,D,C]
    one_hot = one_hot.squeeze(1).permute(0, 4, 1, 2, 3)  # [1,C,H,W,D]
    return one_hot




def dice_multi_class(preds, targets, class_num, class_count):
    smooth = 1e-5
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:] #A CHECKER
    if len(labels) != 15 :
        print(labels)
       
    dices = np.zeros(class_num)
    for label in labels:
        
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices[label -1] = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        class_count[label -1] +=1
    return np.mean(dices), dices, class_count


data_root = "/home/friot/MedSAM2/MedSAM2/amostest/"

#récupérer la liste des images
scan_list = os.listdir(os.path.join(data_root, "spacing075-SURF/maskPredHR1-3-5HR3-tambouille")) #spacing075-SURF/maskPred3Prompt #faibleRes/maskPredLR3HR3
scan_list.sort()



mask_pred = []
gt_files = []

#récupérer les fichiers images et les masques de vérité
#les fichiers de gt et les images doivent porter les memes noms au début 
for scan in scan_list :
    pred = os.path.join(data_root, "spacing075-SURF/maskPredHR1-3-5HR3-tambouille", scan)
    mask_pred.append(pred)
    

    gt_files.append(os.path.join(data_root, "labelsTr", scan))
 
dice_score = []
nsds = []
class_num = 15
class_count = np.zeros(class_num)
for i in range(0,len(gt_files)) :
    
    gt3D_image = sitk.ReadImage(gt_files[i])
    gt3D = sitk.GetArrayFromImage(gt3D_image)
    maskPred = sitk.ReadImage(mask_pred[i])
    spacing = maskPred.GetSpacing()
    spacing = spacing[::-1]
    print(spacing)
    maskPred = sitk.GetArrayFromImage(maskPred)
    

    mean_dice, dices, class_count = dice_multi_class(maskPred, gt3D, class_num, class_count)
    if len(dice_score) == 0 :
            dice_score = [dices]
    else :
        dice_score.append(dices)
    """
    maskPred = to_one_hot(maskPred, class_num=16)
    print(maskPred.shape)
    gt3D = to_one_hot(gt3D, class_num=16)
    nsd = compute_surface_dice(y_pred=y_pred, y=y_true,tolerance=1.0,spacing=spacing,include_background=False,reduction="none")
    
    if len(nsds) == 0 :
        nsds = nsd
    else :
        nsds.append(nsd)
    #print(dice_score)
    """
    print(f"{gt_files[i]} : {np.round(dices,2)} / {np.round(mean_dice, 2)}")
    print(f"{gt_files[i]} : {np.round(nsd,2)}")


dice_score = np.stack(dice_score)
dice_score = np.ma.masked_equal(dice_score, 0)
meanDice = np.round(dice_score.sum(axis=0)/class_count, 3)
print(f"Mean dice score {meanDice}, {np.mean(meanDice)}")
ecart_type = np.ma.std(dice_score, axis=0).filled(np.nan)
print(f"Variance dice score: {np.round(ecart_type,2)}")
"""
nsds = np.stack(nsds)
meanNSDs = np.round(nsds.sum(axis=0)/class_count,3)
print(f"Mean NSD score {meanNSDs}, {np.mean(meanNSDs)}")
"""