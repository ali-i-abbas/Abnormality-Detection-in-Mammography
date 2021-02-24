
import numpy as np
import pandas as pd
import os, errno
import re
import shutil
from shutil import move
import pydicom as dicom
import cv2
import PIL # optional

dataPath = "CBIS-DDSM"
trainPath = "CBIS_png/Train"
testPath = "CBIS_png/Test"
try:
    os.makedirs(trainPath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

"""## Rename and Move Train  files"""

#Rename train Calc/Mass ROI  files and move them to a single directory
def rename_and_move_files (path, origin_dir, dest_dir, filter):

    try:
        os.makedirs(dest_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    directories = [d for d in os.listdir(dataPath) if origin_dir in d]
    
    for directory in directories: 
        subdirs = os.listdir(path + "/" + directory)
        
        
        for subdir in subdirs:
            subsubdirs = os.listdir(path + "/" + directory + "/" + subdir)
            
            for subsubdir in subsubdirs:
                if filter not in subsubdir: continue
                files = os.listdir(path + "/" + directory + "/" + subdir + "/" + subsubdir)

                for file in files:
                    
                    patient_id = str(re.findall("_(P_[\d]+)_", directory))[2:-2]
                    image_side = str(re.findall("_(LEFT|RIGHT)_", directory))[2:-2]
                    image_type = str(re.findall("(CC|MLO)", directory))[2:-2]

                    name = os.path.join(path, directory, subdir, subsubdir, file)
                    image_name = os.path.join(dest_dir, patient_id + "_" + image_side + "_" + image_type + ".png")

                    ds = dicom.dcmread(name)
                    pixel_array_numpy = ds.pixel_array
                    cv2.imwrite(image_name, pixel_array_numpy)

                    os.remove(name)



rename_and_move_files(dataPath, origin_dir = "Mass-Training", dest_dir = trainPath + "/Mass_full_imgs", filter= 'full mammogram images')
rename_and_move_files(dataPath, origin_dir = "Calc-Training", dest_dir = trainPath + "/Calc_full_imgs", filter= 'full mammogram images')

"""## Rename and Move Test files"""

rename_and_move_files(dataPath, origin_dir = "Mass-Test", dest_dir = testPath + "/Mass_full_imgs", filter= 'full mammogram images')
rename_and_move_files(dataPath, origin_dir = "Calc-Test", dest_dir = testPath + "/Calc_full_imgs", filter= 'full mammogram images')


"""## Rename and Move Train Mask  files"""

#Rename train Calc/Mass Mask  files and move them to a single directory
def rename_and_move_files (path, origin_dir, dest_dir, filter):

    try:
        os.makedirs(dest_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    directories = [d for d in os.listdir(dataPath) if origin_dir in d]
    
    for directory in directories: 
        subdirs = os.listdir(path + "/" + directory)
        
        i = 1
        for subdir in subdirs:
            subsubdirs = os.listdir(path + "/" + directory + "/" + subdir)
            
            for subsubdir in subsubdirs:
                if filter not in subsubdir: continue
                files = os.listdir(path + "/" + directory + "/" + subdir + "/" + subsubdir)


                j = 1
                for file in files:

                    name = os.path.join(path, directory, subdir, subsubdir, file)

                    ds = dicom.dcmread(name)

                    if ds.get_item('SeriesDescription') != None:
                        if 'cropped images' in str(ds.get_item('SeriesDescription').value):
                            continue

                    patient_id = str(re.findall("_(P_[\d]+)_", directory))[2:-2]
                    image_side = str(re.findall("_(LEFT|RIGHT)_", directory))[2:-2]
                    image_type = str(re.findall("(CC|MLO)", directory))[2:-2]

                    image_name = os.path.join(dest_dir, patient_id + "_" + image_side + "_" + image_type + "_" + str(i) + str(j) + ".png")

                    pixel_array_numpy = ds.pixel_array
                    cv2.imwrite(image_name, pixel_array_numpy)

                    os.remove(name)

                    i += 1
                    j += 1


rename_and_move_files(dataPath, origin_dir = "Mass-Training", dest_dir = trainPath + "/Mass_MASK_imgs", filter= 'ROI mask images')
rename_and_move_files(dataPath, origin_dir = "Calc-Training", dest_dir = trainPath + "/Calc_MASK_imgs", filter= 'ROI mask images')


"""## Rename and Move Test Mask files"""

rename_and_move_files(dataPath, origin_dir = "Mass-Test", dest_dir = testPath + "/Mass_MASK_imgs", filter= 'ROI mask images')
rename_and_move_files(dataPath, origin_dir = "Calc-Test", dest_dir = testPath + "/Calc_MASK_imgs", filter= 'ROI mask images')



