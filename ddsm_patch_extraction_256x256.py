
# Commented out IPython magic to ensure Python compatibility.
import errno
import random
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
import PIL
from tqdm import tqdm
from PIL import Image, ImageMath
from img_processing_256 import random_flip_img_train

# %pylab inline

def get_png_files (src, dst):
    for root, subdir, files in os.walk(src):
        for file in files:
            ext = os.path.splitext(file)[-1]
            if ext == '.png':
                shutil.move(os.path.join(root, file), dst)

if os.path.isdir("Normal/normal_07"):
    shutil.move("Normal/normal_07", "Normal/Normal07")
if os.path.isdir("Normal/normal_08"):
    shutil.move("Normal/normal_08", "Normal/Normal08")
if os.path.isdir("Normal/normal_09"):
    shutil.move("Normal/normal_09", "Normal/Normal09")
if os.path.isdir("Normal/normal_10"):
    shutil.move("Normal/normal_10", "Normal/Normal10")
if os.path.isdir("Normal/normal_11"):
    shutil.move("Normal/normal_11", "Normal/Normal11")
if os.path.isdir("Normal/normal_12"):
    shutil.move("Normal/normal_12", "Normal/Normal12")

try:
    os.makedirs("Processed_norm_256")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs("Normal/Howtek/Normal07")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs("Normal/Howtek/Normal08")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs("Normal/Howtek/Normal11")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs("Normal/Howtek/Normal12")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs("Normal/Lumysis/Normal09")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs("Normal/Lumysis/Normal10")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

"""## Howtek Scanner - DDSM Normal"""

#Scanner type: Howteck. 
folder_name0 = 'Normal07'
folder_name1 = 'Normal08'
folder_name2 = 'Normal11'
folder_name3 = 'Normal12'

folders_name = [folder_name0, folder_name1, folder_name2, folder_name3]

for folder_name in folders_name:
    src = 'Normal/'+folder_name
    dst = 'Normal/Howtek/'+folder_name
    get_png_files(src,dst)
    shutil.rmtree(src, ignore_errors=True)


"""## Lumysis Scanner - DDSM Normal"""

#Scanner type: Lumysis
folder_name0 = 'Normal09'
folder_name1 = 'Normal10'

folders_name = [folder_name0, folder_name1]

for folder_name in folders_name:
    src = 'Normal/'+folder_name
    dst = 'Normal/Lumysis/'+folder_name
    get_png_files(src,dst)
    shutil.rmtree(src, ignore_errors=True)


"""## Suppress Artifacts"""

ori_img = cv2.imread('Normal/Howtek/Normal11/A_1964_1.LEFT_MLO.png',
          cv2.IMREAD_GRAYSCALE)

threshold = 18  # from Nagi thesis. <<= para to tune!
_, binary_img = cv2.threshold(ori_img, threshold, 
                                maxval=255, type=cv2.THRESH_BINARY)
fig,axes = plt.subplots(1, 2)
fig.set_size_inches([12, 9])
#res = hstack((mammo_med_blurred, mammo_binary))
axes[0].imshow(ori_img, cmap='gray')
axes[1].imshow(binary_img, cmap='gray')

plt.show(block=False)
plt.pause(0.001)

def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
                       smooth_boundary=False, kernel_size=15):
    '''Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin(2D array): 2D numpy array of binary image.
        lab_val([int]): integer value used for the label of the largest 
                        object. Default is 255.
        fill_holes([boolean]): whether fill the holes inside the largest 
                               object or not. Default is false.
        smooth_boundary([boolean]): whether smooth the boundary of the 
                                    largest object using morphological 
                                    opening or not. Default is false.
        kernel_size([int]): the size of the kernel used for morphological 
                            operation.
    '''
    n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(
        img_bin, connectivity=8, ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)
        
    return largest_mask

mask_img = select_largest_obj(binary_img, lab_val=255, 
                                       fill_holes=True, 
                                       smooth_boundary=True, kernel_size=15)  # <<= para to tune!
filtered_img = cv2.bitwise_and(ori_img, mask_img)
fig,axes = plt.subplots(1, 3)
fig.set_size_inches([18, 9])
axes[0].imshow(ori_img, cmap='gray')
axes[1].imshow(mask_img, cmap='gray')
axes[2].imshow(filtered_img, cmap='gray')

plt.show(block=False)
plt.pause(0.001)

def normal_patch_slices(src, upper_thresh=0, lower_thresh=0, mean_thresh=0, stride=200):
    files = os.listdir(src)
    patch_slices = []
    patch_labels = []
    patch_FileNames = []
    
    i = 0
    for file in tqdm(files):
        
        file_name = file
        
        if True:
            print(i, "-", file)
        i += 1
        
        """
        suppress artifacts in image   
        """       
        ori_img=cv2.imread(os.path.join(src, file),cv2.IMREAD_GRAYSCALE)
        
        thresh = 18  # from Nagi thesis. <<= para to tune!
        _, binary_img = cv2.threshold(ori_img, thresh, maxval=255, type=cv2.THRESH_BINARY)
        
        mask_img = select_largest_obj(binary_img, lab_val=255, 
                                       fill_holes=True, 
                                       smooth_boundary=True, kernel_size=15)  # <<= para to tune!
        
        processed_img = cv2.bitwise_and(ori_img, mask_img)
        
        """
        adjust image size and create image slices
        """
        h, w = processed_img.shape
        hmargin = int(h * 0.07)
        wmargin = int(w * 0.07)
        
        img = processed_img[hmargin:h-hmargin, wmargin:w-wmargin]
    
        """
        create 256x256 tiles
        """
        size = 512
        
        tiles = [img[x:x+size,y:y+size] for x in range(0,img.shape[0],stride) for y in range(0,img.shape[1],stride)]
        
        
        """
        filter tiles
        """
        filtered_tiles = []
    
        for i in range(len(tiles)):
            if tiles[i].shape == (size,size):
                if (np.sum(np.sum(tiles[i] >= 225)) < 100) and (np.sum(np.sum(tiles[i] <= 20)) <= 50000):  #filter abnormal tiles
                    if np.mean(tiles[i]) >= mean_thresh: # make sure tile has stuff in it
                        if np.var(tiles[i]) <= upper_thresh: # make sure the tile contains image and not mostly empty space
                            if np.var(tiles[i]) >= lower_thresh:
                                cropped_img = np.array(Image.fromarray(tiles[i]).resize((256,256))) # size the tile down to 256x256
                                
                                filtered_tiles.append(random_flip_img_train(cropped_img.reshape(256,256,1)))
        
        for tile in filtered_tiles:
            patch_slices.append(tile)
            patch_labels.append("NORMAL")
            patch_FileNames.append(file_name)
        
    assert(len(patch_slices) == len(patch_labels))
    
    return np.array(patch_slices), np.array(patch_labels), np.array(patch_FileNames)

"""## Howtek Patches"""

#Howteck Normal07
src = 'Normal/Howtek/Normal07/'
norm07_patches, norm07_labels, norm07_FNs = normal_patch_slices(src, upper_thresh=10000, lower_thresh=10, mean_thresh=30, stride=300)

print("Howtek Normal07 Patches:", len(norm07_patches))
print("Howtek Normal07 Labels:", len(norm07_labels))
print("Howtek Normal07 File Names:", len(norm07_FNs))

norm07_FNs

N = 20
idxs = random.sample(range(len(norm07_patches)), k=N)
plt.figure(figsize=(12,12))
for i, idx in enumerate(idxs):
    plt.subplot(5,4,i+1)
    plt.imshow(norm07_patches[i].reshape(256, 256), cmap ='gist_heat')
    plt.title("Mean:" + str(round(np.mean(norm07_patches[i]),3)) + " | Var:" + str(round(np.var(norm07_patches[i]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

#8-bit patch images
norm07_patches = norm07_patches.astype(np.uint8)

np.save(os.path.join("Processed_norm_256", "howtek_patches_norm07.npy"), norm07_patches)
np.save(os.path.join("Processed_norm_256", "howtek_labels_norm07.npy"), norm07_labels)
np.save(os.path.join("Processed_norm_256", "howtek_FileNames_norm07.npy"), norm07_FNs)

#Howteck Normal08
src = 'Normal/Howtek/Normal08/'
norm08_patches, norm08_labels, norm08_FNs = normal_patch_slices(src, upper_thresh=10000, lower_thresh=10, mean_thresh=30, stride=300)

print("Howtek Normal08 Patches:", len(norm08_patches))
print("Howtek Normal08 Labels:", len(norm08_labels))
print("Howtek Normal08 FileNames:", len(norm08_FNs))

N = 20
idxs = random.sample(range(len(norm08_patches)), k=N)
plt.figure(figsize=(12,12))
for i, idx in enumerate(idxs):
    plt.subplot(5,4,i+1)
    plt.imshow(norm08_patches[i].reshape(256,256), cmap ='gist_heat')
    plt.title("Mean:" + str(round(np.mean(norm08_patches[i]),3)) + " | Var:" + str(round(np.var(norm08_patches[i]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

#8-bit
norm08_patches = norm08_patches.astype(np.uint8)

np.save(os.path.join("Processed_norm_256", "howtek_patches_norm08.npy"), norm08_patches)
np.save(os.path.join("Processed_norm_256", "howtek_labels_norm08.npy"), norm08_labels)
np.save(os.path.join("Processed_norm_256", "howtek_FileNames_norm08.npy"), norm08_FNs)

#Howteck Normal11
src = 'Normal/Howtek/Normal11/'
norm11_patches, norm11_labels, norm11_FNs = normal_patch_slices(src, upper_thresh=10000, lower_thresh=10, mean_thresh=30, stride=350)

print("Howtek Normal11 Patches:", len(norm11_patches))
print("Howtek Normal11 Labels:", len(norm11_labels))
print("Howtek Normal11 File Names:", len(norm11_FNs))

N = 16
idxs = random.sample(range(len(norm11_patches)), k=N)
plt.figure(figsize=(12,12))
for i, idx in enumerate(idxs):
    plt.subplot(5,4,i+1)
    plt.imshow(norm11_patches[i].reshape(256,256), cmap ='gist_heat')
    plt.title("Mean:" + str(round(np.mean(norm11_patches[i]),3)) + " | Var:" + str(round(np.var(norm11_patches[i]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

#8-bit images
norm11_patches = norm11_patches.astype(np.uint8)

np.save(os.path.join("Processed_norm_256", "howtek_patches_norm11.npy"), norm11_patches)
np.save(os.path.join("Processed_norm_256", "howtek_labels_norm11.npy"), norm11_labels)
np.save(os.path.join("Processed_norm_256", "howtek_FileNames_norm08.npy"), norm11_FNs)

#Howteck Normal12
src = 'Normal/Howtek/Normal12/'
norm12_patches, norm12_labels, norm12_FNs = normal_patch_slices(src, upper_thresh=10000, lower_thresh=10, mean_thresh=30, stride=300)

print("Howtek Normal12 Patches:", len(norm12_patches))
print("Howtek Normal12 Labels:", len(norm12_labels))
print("Howtek Normal12 Labels:", len(norm12_FNs))

N = 20
idxs = random.sample(range(len(norm11_patches)), k=N)
plt.figure(figsize=(12,12))
for i, idx in enumerate(idxs):
    plt.subplot(5,4,i+1)
    plt.imshow(norm11_patches[i].reshape(256, 256), cmap ='gist_heat')
    plt.title("Mean:" + str(round(np.mean(norm11_patches[i]),3)) + " | Var:" + str(round(np.var(norm11_patches[i]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

#8-bit images
norm12_patches = norm12_patches.astype(np.uint8)

np.save(os.path.join("Processed_norm_256", "howtek_patches_norm12.npy"), norm12_patches)
np.save(os.path.join("Processed_norm_256", "howtek_labels_norm12.npy"), norm12_labels)
np.save(os.path.join("Processed_norm_256", "howtek_labels_norm12.npy"), norm12_FNs)

"""**Merge Howtek**"""

howtek_patches = np.concatenate([norm07_patches, norm08_patches, norm11_patches, norm12_patches], axis = 0)
howtek_labels = np.concatenate([norm07_labels, norm08_labels, norm11_labels, norm12_labels], axis = 0)
howtek_FNs = np.concatenate([norm07_FNs, norm08_FNs, norm11_FNs, norm12_FNs], axis = 0)

np.save(os.path.join("Processed_norm_256", "howtek_patches_all.npy"), howtek_patches)
np.save(os.path.join("Processed_norm_256", "howtek_labels_all.npy"), howtek_labels)
np.save(os.path.join("Processed_norm_256", "howtek_FileNames_all.npy"), howtek_FNs)

"""## Lumysis Patches"""

#Lumysis Normal09
src = 'Normal/Lumysis/Normal09/'
norm09_patches, norm09_labels, norm09_FNs = normal_patch_slices(src, upper_thresh=10000, lower_thresh=10, mean_thresh=28, stride=300)

print("Lumysis Normal09 Patches:", len(norm09_patches))
print("Lumysis Normal09 Labels:", len(norm09_labels))
print("Lumysis Normal09 FileNames:", len(norm09_FNs))

N = 16
idxs = random.sample(range(len(norm09_patches)), k=N)
plt.figure(figsize=(12,12))
for i, idx in enumerate(idxs):
    plt.subplot(5,4,i+1)
    plt.imshow(norm09_patches[i].reshape(256, 256), cmap ='gist_heat')
    plt.title("Mean:" + str(round(np.mean(norm09_patches[i]),3)) + " | Var:" + str(round(np.var(norm09_patches[i]),3)))
    plt.tight_layout() 
plt.show(block=False)
plt.pause(0.001)

#normalize patch images
norm09_patches = norm09_patches.astype(np.uint8)

np.save(os.path.join("Processed_norm_256", "lumysis_patches_norm09.npy"), norm09_patches)
np.save(os.path.join("Processed_norm_256", "lumysis_labels_norm09.npy"), norm09_labels)
np.save(os.path.join("Processed_norm_256", "lumysis_FileNames_norm09.npy"), norm09_FNs)

#Howteck Normal10
src = 'Normal/Lumysis/Normal10/'
norm10_patches, norm10_labels, norm10_FNs = normal_patch_slices(src, upper_thresh=10000, lower_thresh=10, mean_thresh=28, stride=300)

print("Lumysis Normal10 Patches:", len(norm10_patches))
print("Lumysis Normal10 Labels:", len(norm10_labels))
print("Lumysis Normal10 Labels:", len(norm10_FNs))

N = 16
idxs = random.sample(range(len(norm10_patches)), k=N)
plt.figure(figsize=(12,12))
for i, idx in enumerate(idxs):
    plt.subplot(4,4,i+1)
    plt.imshow(norm10_patches[i].reshape(256, 256), cmap ='gist_heat')
    plt.title("Mean:" + str(round(np.mean(norm10_patches[i]),3)) + " | Var:" + str(round(np.var(norm10_patches[i]),3)))
    plt.tight_layout() 
plt.show(block=False)
plt.pause(0.001)

#Convert to 8bit patch images
norm10_patches = norm10_patches.astype(np.uint8)

np.save(os.path.join("Processed_norm_256", "lumysis_patches_norm10.npy"), norm10_patches)
np.save(os.path.join("Processed_norm_256", "lumysis_labels_norm10.npy"), norm10_labels)
np.save(os.path.join("Processed_norm_256", "lumysis_FileNames_norm10.npy"), norm10_FNs)

"""**All Lumysis**"""

lumysis_patches = np.concatenate([norm09_patches, norm10_patches], axis = 0)
lumysis_labels = np.concatenate([norm09_labels, norm10_labels], axis = 0)
lumysis_FNs = np.concatenate([norm09_labels, norm10_labels], axis = 0)

np.save(os.path.join("Processed_norm_256", "lumysis_patches_all.npy"), lumysis_patches)
np.save(os.path.join("Processed_norm_256", "lumysis_labels_all.npy"), lumysis_labels)
np.save(os.path.join("Processed_norm_256", "lumysis_FileNames_all.npy"), lumysis_FNs)



plt.show()


