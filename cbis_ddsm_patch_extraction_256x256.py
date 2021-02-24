
# Commented out IPython magic to ensure Python compatibility.
import errno
import re
import numpy as np
import pandas as pd
import os
import PIL
import random
import shutil
import matplotlib.pyplot as plt
import PIL
import sys
import cv2
from networkx.tests.test_all_random_functions import progress
from tqdm import tqdm
from PIL import Image, ImageMath

from skimage.transform import rescale, resize, downscale_local_mean
from img_processing_256 import mask_img, rename, random_flip_img_train, crop_img
# %matplotlib inline

def create_patches(mask_dir, img_dir, Lbls, size=256, debug=True):
    patch_list = []
    Lbl_list = []
    FN_list = []
    roi_sizes = []
    full_size = 512
    masks = os.listdir(mask_dir)
    counter = 0
    if debug is None:
        progress(counter, len(masks), 'WORKING')
    for mask in tqdm(masks):
        counter += 1
        if debug is None:
            progress(counter, len(masks), mask)    
        base_img_file = rename(mask)
        try:
            full_img = PIL.Image.open(img_dir + "/" + base_img_file + '.png')
        except:
            try:
                full_img = PIL.Image.open(img_dir + "/" + base_img_file + '000000.png')
            except:
                try:
                    full_img = PIL.Image.open(img_dir + "/" + base_img_file + '000001.png')
                except:
                    print("Error FileNotFound:", base_img_file)
                    continue
        try:
            Lbl = Lbls.loc[base_img_file + ".png"]['Class']
        except:
            print("Error LabelNotFound", base_img_file)
            continue
        
        full_img_arr = np.array(full_img)
        ctr_row, ctr_col, too_big, full_img_arr, mask_size = mask_img(mask_dir + "/" + mask,full_img_arr, half=False,
                                                                         output=debug)
        img_h, img_w = full_img_arr.shape
        try:
            mask_H = mask_size[0]
            mask_W = mask_size[1]
            roi_size = np.max([mask_H, mask_W])
            if debug:
                print("Mask", mask, " Height:", mask_H, "Width:", mask_W)
        except:
            print("Mask Size Error:", mask_size, "for", mask)
        # Record roi size for DDSM image crop
        roi_sizes.append(roi_size)
        if (ctr_row == 0) and (ctr_col == 0):
            print("Error, skipping", mask)
            continue
        """
        Extract the ROI depending on it's size
        If the ROI is smaller than a slice extract it with some padding
        """
        if roi_size < full_size:
            if debug:
                print("ROI small", mask)
            ## Make sure the size of the ROI is at least as big as a tile will be
            adj_mask_H = int(np.max([full_size * 1.4, mask_H]))
            adj_mask_W = int(np.max([full_size * 1.4, mask_W]))
            ## Extract the full ROI with 20% padding on either side
            start_row = int(np.max([ctr_row - (adj_mask_H // 2), 0]))
            end_row = start_row + adj_mask_H
            if end_row > img_h:
                end_row = img_h
                start_row = img_h - adj_mask_H
            start_col = int(np.max([ctr_col - (adj_mask_W // 2), 0]))
            end_col = start_col + adj_mask_W
            if end_col > img_w:
                end_col = img_w
                start_col = img_w - adj_mask_W

            # extract the ROI and randomly flip it
            roi_img = random_flip_img_train(full_img_arr[start_row:end_row, start_col:end_col])
        # else extract the ROI with less padding
        else:
            if debug:
                print("ROI Big", mask)
            # padding for the random cropping
            adj_mask_H = int(np.max([full_size * 1.15, mask_H]))
            adj_mask_W = int(np.max([full_size * 1.15, mask_W]))
            start_row = np.max([ctr_row - (adj_mask_H // 2), 0])
            end_row = start_row + adj_mask_H
            if end_row > img_h:
                end_row = img_h
                start_row = img_h - adj_mask_H
            start_col = np.max([ctr_col - (adj_mask_W // 2), 0])
            end_col = start_col + adj_mask_W
            if end_col > img_w:
                end_col = img_w
                start_col = img_w - adj_mask_W
            # extract the ROI and randomly flip it
            roi_img = random_flip_img_train(full_img_arr[start_row:end_row, start_col:end_col])
              
        patch_1 = crop_img(roi_img)
        patch_2 = crop_img(roi_img)
        patch_3 = crop_img(roi_img)
         
        if (patch_1.shape[0] == size) and (patch_1.shape[1] == size):
            patch_list.append(patch_1)
            Lbl_list.append(Lbl)
            FN_list.append(base_img_file + ".png")
                
        if (patch_2.shape[0] == size) and (patch_2.shape[1] == size):
            patch_list.append(patch_2)
            Lbl_list.append(Lbl)
            FN_list.append(base_img_file + ".png")
        
        if (patch_3.shape[0] == size) and (patch_2.shape[1] == size):
            patch_list.append(patch_3)
            Lbl_list.append(Lbl)
            FN_list.append(base_img_file + ".png")
                
    return np.array(patch_list), np.array(Lbl_list), np.array(FN_list), roi_sizes

"""## Train mass data"""

try:
    os.makedirs("Processed_abnorm_256")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

train_labels = pd.read_pickle("train_label.pkl")
train_labels['IMAGE_NAME2'] = train_labels.index
train_labels = train_labels.drop_duplicates(['IMAGE_NAME2'])

## use a copy on the local drive to make testing faster
mask_dir = "CBIS_png/Train/Mass_MASK_imgs"
img_dir = "CBIS_png/Train/Mass_full_imgs"

train_mass_patch, train_mass_Lbl, train_mass_FN, train_mass_roi_size = \
        create_patches(mask_dir, img_dir, Lbls=train_labels, debug=True)

print("Train mass patches shape:", train_mass_patch.shape)
print("Train mass Labels:", len(train_mass_Lbl))
print("Train mass File Name:", len(train_mass_FN))

#calc_img1 = cv2.imread('./CBIS_png/Train/Calc_full_imgs/P_00989_RIGHT_MLO.png', cv2.IMREAD_GRAYSCALE)
#calc_img2 = cv2.imread('./CBIS_png/Train/Calc_full_imgs/P_01033_LEFT_CC.png', cv2.IMREAD_GRAYSCALE)
mass_img1 = cv2.imread('./CBIS_png/Train/Mass_full_imgs/P_00001_LEFT_CC.png', cv2.IMREAD_GRAYSCALE)
mass_mask_img2 = cv2.imread('./CBIS_png/Train/Mass_MASK_imgs/P_00001_LEFT_CC_11.png', cv2.IMREAD_GRAYSCALE)

# Commented out IPython magic to ensure Python compatibility.
# %pylab inline
fig,ax = plt.subplots(1, 3)
fig.set_size_inches([10, 8])
ax[0].imshow(mass_mask_img2, cmap='gray')
ax[1].imshow(mass_img1, cmap='gray')
ax[2].imshow(train_mass_patch[0].reshape(256, 256), cmap='gray')
fig.tight_layout()

plt.imshow(train_mass_patch[0].reshape(256, 256), cmap='gray')

plt.imshow(train_mass_patch[4].reshape(256, 256), cmap='gray')

# random images 
N = 20
idx = random.sample(range(len(train_mass_patch)), k=N)
plt.figure(figsize=(12,12))
for i, j in enumerate(idx):
    plt.subplot(5,4,i+1)
    plt.imshow(train_mass_patch[j].reshape(256, 256), cmap='gist_heat')
    plt.title(train_mass_FN[j] + " - " + str(j)+ "\n" + "Mean:" + str(round(np.mean(train_mass_patch[j]),3)) + " | Var:" + str(round(np.var(train_mass_patch[j]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

print("ROI Mean Size:", np.mean(train_mass_roi_size))
print("ROI Min Size:", np.min(train_mass_roi_size))
print("ROI Max Size:", np.max(train_mass_roi_size))
print("ROI Size Std:", np.std(train_mass_roi_size))

print("ROI Mean Size:", np.mean(train_mass_roi_size))
print("ROI Min Size:", np.min(train_mass_roi_size))
print("ROI Max Size:", np.max(train_mass_roi_size))
print("ROI Size Std:", np.std(train_mass_roi_size))

np.save(os.path.join("Processed_abnorm_256", "train_mass_patch.npy"), train_mass_patch)
np.save(os.path.join("Processed_abnorm_256", "train_mass_Lbl.npy"), np.array(train_mass_Lbl))
np.save(os.path.join("Processed_abnorm_256", "train_mass_FN.npy"), train_mass_FN)
np.save(os.path.join("Processed_abnorm_256", "train_mass_roi_size.npy"), np.array(train_mass_roi_size))

"""## Train calc data"""

train_labels = pd.read_pickle("train_label.pkl")
train_labels['IMAGE_NAME2'] = train_labels.index
train_labels = train_labels.drop_duplicates(['IMAGE_NAME2'])

## use a copy on the local drive to make testing faster
mask_dir = "CBIS_png/Train/Calc_MASK_imgs"
img_dir = "CBIS_png/Train/Calc_full_imgs"

train_calc_patch, train_calc_Lbl, train_calc_FN, train_calc_roi_size = \
        create_patches(mask_dir, img_dir, Lbls=train_labels, debug=True)

print("Train calc patches shape:", train_calc_patch.shape)
print("Train calc Labels:", len(train_calc_Lbl))
print("Train calc File Name:", len(train_calc_FN))

# random images 
N = 20
idx = random.sample(range(len(train_calc_patch)), k=N)
plt.figure(figsize=(12,12))
for i, j in enumerate(idx):
    plt.subplot(5,4,i+1)
    plt.imshow(train_calc_patch[j].reshape(256, 256), cmap='gray')
    plt.title(train_calc_FN[j] + " - " + str(j)+ "\n" + "Mean:" + str(round(np.mean(train_calc_patch[j]),3)) + " | Var:" + str(round(np.var(train_calc_patch[j]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

print("ROI Mean Size:", np.round(np.mean(train_calc_roi_size),2))
print("ROI Min Size:", np.min(train_calc_roi_size))
print("ROI Max Size:", np.max(train_calc_roi_size))
print("ROI Size Std:", np.round(np.std(train_calc_roi_size),2))

np.save(os.path.join("Processed_abnorm_256", "train_calc_patch.npy"), train_calc_patch)
np.save(os.path.join("Processed_abnorm_256", "train_calc_Lbl.npy"), np.array(train_calc_Lbl))
np.save(os.path.join("Processed_abnorm_256", "train_calc_FN.npy"), train_calc_FN)
np.save(os.path.join("Processed_abnorm_256", "train_calc_roi_size.npy"), np.array(train_calc_roi_size))

"""## Test mass data"""

test_labels = pd.read_pickle("test_label.pkl")
test_labels['IMAGE_NAME2'] = test_labels.index
test_labels = test_labels.drop_duplicates(['IMAGE_NAME2'])

## use a copy on the local drive to make testing faster
mask_dir = "CBIS_png/Test/Mass_MASK_imgs"
img_dir = "CBIS_png/Test/Mass_full_imgs"

test_mass_patch, test_mass_Lbl, test_mass_FN, test_mass_roi_size = \
    create_patches(mask_dir, img_dir, Lbls=test_labels, debug=True)

print("test mass patches shape:", test_mass_patch.shape)
print("test mass Labels:", len(test_mass_Lbl))
print("test mass File Name:", len(test_mass_FN))

# random images 
N = 16
idx = random.sample(range(len(test_mass_patch)), k=N)
plt.figure(figsize=(12,12))
for i, j in enumerate(idx):
    plt.subplot(5,4,i+1)
    plt.imshow(test_mass_patch[j].reshape(256, 256), cmap='gray')
    plt.title(test_mass_FN[j] + " - " + str(j)) #+ "\n" + "Mean:" + str(round(np.mean(test_mass_patch[j]),3)) + " | Var:" + str(round(np.var(test_mass_patch[j]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

print("ROI Mean Size:", np.round(np.mean(test_mass_roi_size),2))
print("ROI Min Size:", np.min(test_mass_roi_size))
print("ROI Max Size:", np.max(test_mass_roi_size))
print("ROI Size Std:", np.round(np.std(test_mass_roi_size),2))

np.save(os.path.join("Processed_abnorm_256", "test_mass_patch.npy"), test_mass_patch)
np.save(os.path.join("Processed_abnorm_256", "test_mass_Lbl.npy"), np.array(test_mass_Lbl))
np.save(os.path.join("Processed_abnorm_256", "test_mass_FN.npy"), test_mass_FN)
np.save(os.path.join("Processed_abnorm_256", "test_mass_roi_size.npy"), np.array(test_mass_roi_size))

"""##  Test calc data"""

test_labels = pd.read_pickle("test_label.pkl")
test_labels['IMAGE_NAME2'] = test_labels.index
test_labels = test_labels.drop_duplicates(['IMAGE_NAME2'])

## use a copy on the local drive to make testing faster
mask_dir = "CBIS_png/Test/Calc_MASK_imgs"
img_dir = "CBIS_png/Test/Calc_full_imgs"

test_calc_patch, test_calc_Lbl, test_calc_FN, test_calc_roi_size = \
        create_patches(mask_dir, img_dir, Lbls=test_labels, debug=True)

print("Test calc patches shape:", test_calc_patch.shape)
print("Test calc Labels:", len(test_calc_Lbl))
print("Test calc File Name:", len(test_calc_FN))

# random images 
N = 20
idx = random.sample(range(len(test_calc_patch)), k=N)
plt.figure(figsize=(12,12))
for i, j in enumerate(idx):
    plt.subplot(5,4,i+1)
    plt.imshow(test_calc_patch[j].reshape(256, 256), cmap='gist_heat')
    plt.title(test_calc_FN[j] + " - " + str(j)+ "\n" + "Mean:" + str(round(np.mean(test_calc_patch[j]),3)) + " | Var:" + str(round(np.var(test_calc_patch[j]),3)))
    plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

print("ROI Mean Size:", np.round(np.mean(test_calc_roi_size),2))
print("ROI Min Size:", np.min(test_calc_roi_size))
print("ROI Max Size:", np.max(test_calc_roi_size))
print("ROI Size Std:", np.round(np.std(test_calc_roi_size),2))

np.save(os.path.join("Processed_abnorm_256", "test_calc_patch.npy"), test_calc_patch)
np.save(os.path.join("Processed_abnorm_256", "test_calc_Lbl.npy"), np.array(test_calc_Lbl))
np.save(os.path.join("Processed_abnorm_256", "test_calc_FN.npy"), test_calc_FN)
np.save(os.path.join("Processed_abnorm_256", "test_calc_roi_size.npy"), np.array(test_calc_roi_size))

"""## Merge data"""

train_patch = np.concatenate([train_mass_patch, train_calc_patch], axis=0)
train_Lbl = np.concatenate([train_mass_Lbl, train_calc_Lbl], axis=0)
train_FN = np.concatenate([train_mass_FN, train_calc_FN], axis=0)

test_patch = np.concatenate([test_mass_patch, test_calc_patch], axis=0)
test_Lbl = np.concatenate([test_mass_Lbl, test_calc_Lbl], axis=0)
test_FN = np.concatenate([test_mass_FN, test_calc_FN], axis=0)



print("Train Patches:", train_patch.shape)
print("Train Lables:", train_Lbl.shape)
print("Train File Names:", train_FN.shape)

print("Test Patches:", test_patch.shape)
print("Test Lables:", test_Lbl.shape)
print("Test File Names:", test_FN.shape)

np.save(os.path.join("Processed_abnorm_256", "abnormal_train_patch.npy"), train_patch)
np.save(os.path.join("Processed_abnorm_256", "abnormal_train_Lbl.npy"), train_Lbl)
np.save(os.path.join("Processed_abnorm_256", "abnormal_train_FN.npy"), train_FN)

np.save(os.path.join("Processed_abnorm_256", "abnormal_test_patch.npy"), test_patch)
np.save(os.path.join("Processed_abnorm_256", "abnormal_test_Lbl.npy"), test_Lbl)
np.save(os.path.join("Processed_abnorm_256", "abnormal_test_FN.npy"), test_FN)

# probably not needed?
#all_roi_sizes = np.concatenate([train_mass_roi_size, train_calc_roi_size, test_mass_roi_size, test_calc_roi_size], axis=0)
#np.save(os.path.join("Processed_abnorm_256", "abnormal_all_roi_sizes.npy"), all_roi_sizes)

plt.show()