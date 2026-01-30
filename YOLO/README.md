Dans ce dossier se situe le code pour inférer des modèles YOLO pour la génération de boîtes englobantes avec MedSAM2.

## Inference de YOLO (yolo_inference.py)

Ce code permet de générer un fichier json (3d_bounding_boxes_results.json) contenant 3 boîtes englobantes pour chaques organes de chaque images ainsi que les bornes de propagation (z_min et z_max) générées avec un détecteur pré-entrainé.

Architecture du json : 
```text
[
  {
    "image_name": "/chemin/vers/image_1.nii.gz",
    "organs": {
      "organ_id": {
        "class_name": "nom_organe",
        "z_min": 100,
        "z_max": 200,
        "bboxes": [
          [z_25, [x1, y1, x2, y2]], #altitude boite enlobante inférieure et coordonées de cette boîte englobante
          [z_50, [x1, y1, x2, y2]], #altitude boite enlobante milieu et coordonées de cette boîte englobante
          [z_75, [x1, y1, x2, y2]] #altitude boite enlobante supérieure et coordonées de cette boîte englobante
        ]
      }
    }
  }
]
```
Pour exéctuter ce script, il faut donner/changer la variable MODEL_PATH qui est le chemin du checkpoint du YOLO pré-entrainé.

Dans mon script j'ai une fonction converToLAS car les modèles YOLO que j'ai utilisés ont été entrainé avec des images en format LAS, alors que mes images étaient en format RAS. Je devais donc retourner mes images avant de les inférer. 
De plus, après inférence, je devais à l'inverse convertir les coordonnées obtenues en LAS dans le référentiel RAS. Je conseille d'enlever ces deux étapes et d'entrainer les modèles YOLO dans le même référentiel (RAS ou LAS) que pour l'inférence avec MedSAM2.


## Inference de MedSAM2

Pour installer l'environement de MedSAM2 : 
  https://github.com/bowang-lab/MedSAM2
  

Ensuite, il y a deux scripts permettant d'inférer MedSAM2 en utilisant une ou 3 boîtes englobantes par organes.

Pour exéctuer ces deux scripts, il faut changer plusiuers variables :
  - dataroot : le chemin vers le dossier contenant les images
  - scan_list, path_tr, gt_files : chemins vers les dossiers contenant les images à inférer et leur label
  - checkpoint (de MedSAM2)
  - model_cfg (de MedSAM2)
  - Et tout à la fin quand le masque prédit et enregistré :  sitk.WriteImage(sitk_mask, mask_path.replace('labelsVa', f"maskPred-yolov11n3P") ) : si dans mask_path il y a pas labelsVa, les masques de vérité sont écrasés
