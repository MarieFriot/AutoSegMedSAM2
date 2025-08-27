import nibabel as nib
from os import listdir
from os.path import abspath, dirname, isdir, join, normpath
import argparse
def flipAndSaveToRAS( filename ):
    
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
        img_conv = nib.Nifti1Image(img_data.astype(flippedImage.header.get_data_dtype()), flippedImage.affine, flippedImage.header)

        #Set Qcode to 1 that the Qform matrix can be used into the further processing
        img_conv.header['qform_code'] = 1
        
        #Save the flipped image
        nib.save(img_conv, filename )

        print("The new orientation is now : ", NewOrientation)

def size(filename, l512, l768) :
    img = nib.load( filename )
    header = img.header
    h, w, d= header.get_data_shape()
    """
    if (h,w) != (512,512) :
        l512.append(filename)
        print(f"Image {filename} de taille {(d,h,w)}")
    else : 
        l768.append(filename)
    
    """
    print(d)
    l512.append(d)
    return l512, l768
    


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Traitement des images RAS")
    parser.add_argument("targetImagesDir", help="Chemin vers le dossier contenant les images à traiter")
    args = parser.parse_args()
    targetImagesDir = args.targetImagesDir
    l512 = []
    l768 = []
    flipAndSaveToRAS("/home/friot/MedSAM2/MedSAM2/amostest/labelsTr/amos_0309.nii.gz")
    """
    for f in sorted(listdir(targetImagesDir)):
        for ext in [".nii.gz", ".mhd", ".csv.gz"]:
            if f.endswith(ext):
                flipAndSaveToRAS(abspath(join(targetImagesDir, f)))
    """           
               