
import nibabel as nib
import os
from os import listdir
from os.path import abspath, dirname, isdir, join, normpath
import tempfile
import time
import numpy as np
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
def separate():
	print( "******************************************************" )


def flipToRAS( filename ):
    
    #Recover the image object
    imageObj = nib.load( filename )
    
    #Get the current orientation
    CurrentOrientation = nib.aff2axcodes(imageObj.affine)
    print("The current orientation is : ", CurrentOrientation)
    
    #Check if the current orientation is already RAS+
    if CurrentOrientation == ('R', 'A', 'S') :
        
        print("Image already recorded into the RAS+ orientation, nothing to do")
        return filename
        
    else :
        #Flip the image to RAS
        flippedImage = nib.as_closest_canonical(imageObj)
                
        ##Check the new orientation
        NewOrientation = nib.aff2axcodes(flippedImage.affine)
        img_data = flippedImage.get_fdata()
        img_conv = nib.Nifti1Image(img_data.astype(flippedImage.headr.get_data_dtype()), flippedImage.affine, flippedImage.header)

        #Set Qcode to 1 that the Qform matrix can be used into the further processing
        img_conv.header['qform_code'] = 1
        nib.save(img_conv, filename )

        print("The new orientation is now : ", NewOrientation)
        return filename
       
       
       



def execute(cmd) : 
    start_time = time.time()
    separate()
    print("Executing : " + cmd)
    code = os.system(cmd)
    print( "Command : " + cmd)
    print( "Executed in" + str(round(time.time()- start_time)) + "s")
    print("Exit code :" + str(code))
    if code :  raise OSError(code)

reference_mask = []
transRef = []

dummyFile = "/home/friot/AutoSegMedSAM2/FROG-master/output/dummy.mhd"
transformed_folder= "/home/friot/AutoSegMedSAM2/FROG-master/output/"

outputMasks = "/home/friot/AutoSegMedSAM2/MedSAM2/amos/scoreMap"




imagesRgDir = "/home/friot/AutoSegMedSAM2/MedSAM2/amos/imagesRg" #images qui ont servi à faire l'atlas
labelsRgDir = imagesRgDir.replace('imagesRg', 'labelsRg')
imagesRgPaths = []
imagesRg = []

for i, f in enumerate(sorted(listdir(imagesRgDir))):
    for ext in [".nii.gz", ".mhd", ".csv.gz"]:
        if f.endswith(ext):
            reference_mask.append(abspath(join(labelsRgDir, f)))
            transRef.append(i)



transformBin = "/home/friot/AutoSegMedSAM2/FROG-master/bin/VolumeTransform"


#Déplace les masques de référence dans l'espace commun
for i in range(len(reference_mask)) :
    mask_path = reference_mask[i]
    execute( " ".join( [ transformBin, mask_path, dummyFile, "-t " + "/home/friot/AutoSegMedSAM2/FROG-master/output/transforms/"+f"{transRef[i]}"+".json" + " -o " + join(transformed_folder, f"transformed_mask{i}.nii.gz")  + " -i nearestneighbour" ] ) )



imagesTrDir = "/home/friot/AutoSegMedSAM2/MedSAM2/amos/imagesTr"
imagesTrPaths = []
imagesTr = []

for f in sorted(listdir(imagesTrDir)):
    for ext in [".nii.gz", ".mhd", ".csv.gz"]:
        if f.endswith(ext):
            imagesTrPaths.append(abspath(join(imagesTrDir, f)))
            imagesTr.append(f)

MAX_WORKERS = 1
print(imagesTr)

duration = []

for i in range(len(imagesTr)) :
    start_time = time.time()

    transformed_mask_target = join(outputMasks, imagesTr[i]) #chemin ou doivent être enregistrée les cartes de scores

    cmds = []

    #déplace les masques de référence de l'espace commun à l'espace de l'image cible
    for k in range(len(reference_mask)) :
        temp_mask_path = join(transformed_folder, f"transformed_maskTemp{k}.nii.gz")
        cmd =  " ".join( [ transformBin, join(transformed_folder, f"transformed_mask{k}.nii.gz") , imagesTrPaths[i], " -ti " + "/home/friot/AutoSegMedSAM2/FROG-master/transformsR/"+f"{i}"+".json" + " -o " + temp_mask_path+ " -i nearestneighbour" ] ) 
        execute(cmd)
    


    mask = []
    #supperpose les masques transposés
    for k in range(len(reference_mask)) :
        temp_mask_path = join(transformed_folder, f"transformed_maskTemp{k}.nii.gz")
        loaded_mask = nib.load(temp_mask_path)
        mask.append(loaded_mask.get_fdata().astype(np.uint8))

    mask = np.stack(mask, axis = 0)
    mask = nib.Nifti1Image(mask, loaded_mask.affine)
    nib.save(mask, transformed_mask_target) #enrefistre les cartes de scores non seuillées dans le dossier transformed_mask_target
    duration.append(round(time.time()-start_time))
 
print(duration)
print(np.mean(duration))
print(np.std(duration))
