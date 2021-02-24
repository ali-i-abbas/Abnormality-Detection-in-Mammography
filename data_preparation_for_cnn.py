
# Commented out IPython magic to ensure Python compatibility.
import errno
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# %matplotlib inline
import gc

try:
    os.makedirs("Label")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs("Data/256")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

"""## Load Data

**CBIS-DDSM Data (Abnormal Images)**
"""

#train data
CBIS_train_patches = np.load(os.path.join("./Processed_abnorm_256", "abnormal_train_patch.npy" ))
CBIS_train_labels = np.load(os.path.join("./Processed_abnorm_256", "abnormal_train_Lbl.npy" ))
CBIS_train_FNs = np.load(os.path.join("./Processed_abnorm_256", "abnormal_train_FN.npy" ))

#test data
CBIS_test_patches = np.load(os.path.join("./Processed_abnorm_256", "abnormal_test_patch.npy" ))
CBIS_test_labels = np.load(os.path.join("./Processed_abnorm_256", "abnormal_test_Lbl.npy" ))
CBIS_test_FNs = np.load(os.path.join("./Processed_abnorm_256", "abnormal_test_FN.npy" ))



print("Abnaormal train Patches:", CBIS_train_patches.shape)
print("Abnaormal train Labels:", CBIS_train_labels.shape)
print("Abnaormal train File Names:", CBIS_train_FNs.shape)
print("\n")
print("Abnaormal test Patches:", CBIS_test_patches.shape)
print("Abnaormal test Labels:", CBIS_test_labels.shape)
print("Abnaormal test File Names:", CBIS_test_FNs.shape)

#combine train and test data 

CBIS_all_patches = np.concatenate([CBIS_train_patches, CBIS_test_patches], axis=0)
del CBIS_train_patches
del CBIS_test_patches
gc.collect()
CBIS_all_labels = np.concatenate([CBIS_train_labels, CBIS_test_labels], axis=0)
del CBIS_train_labels
del CBIS_test_labels
gc.collect()
CBIS_all_FNs = np.concatenate([CBIS_train_FNs, CBIS_test_FNs], axis=0)
del CBIS_train_FNs
del CBIS_test_FNs
gc.collect()

CBIS_all_patches, CBIS_all_labels, CBIS_all_FNs = \
shuffle(CBIS_all_patches, CBIS_all_labels, CBIS_all_FNs, random_state=19510705)

#split the combined data into train and test
train_patches, test_patches, train_labels, test_labels, train_FNs, test_FNs = \
train_test_split(CBIS_all_patches, CBIS_all_labels, CBIS_all_FNs, test_size = 0.183565, random_state=19430727)
del CBIS_all_patches
del CBIS_all_labels
del CBIS_all_FNs
gc.collect()

"""**DDSM Data (Normal Images)**"""

#Howtek data
howtek_patches = np.load(os.path.join("./Processed_norm_256", "howtek_patches_all.npy" ))
howtek_labels = np.load(os.path.join("./Processed_norm_256", "howtek_labels_all.npy" ))
howtek_FNs = np.load(os.path.join("./Processed_norm_256", "howtek_FileNames_all.npy" ))

#Lumysis data
lumysis_patches = np.load(os.path.join("./Processed_norm_256", "howtek_patches_all.npy" ))
lumysis_labels = np.load(os.path.join("./Processed_norm_256", "howtek_labels_all.npy" ))
lumysis_FNs = np.load(os.path.join("./Processed_norm_256", "howtek_FileNames_all.npy" ))

#combined normal data 
normal_patches = np.concatenate([howtek_patches, lumysis_patches], axis = 0)
del howtek_patches
del lumysis_patches
gc.collect()
normal_labels = np.concatenate([howtek_labels, lumysis_labels], axis = 0)
del howtek_labels
del lumysis_labels
gc.collect()
normal_FNs = np.concatenate([howtek_FNs, lumysis_FNs], axis = 0)
del howtek_FNs
del lumysis_FNs
gc.collect()

# print("Normal Patches:", normal_patches.shape)
# print("Normal Labels:", normal_labels.shape)
# print("Normal File Names:", normal_FNs.shape)

#Shuffle and split DDSM into train and test dataset
normal_patches, normal_labels, normal_FNs = \
shuffle(normal_patches, normal_labels, normal_FNs, random_state=20170301)

print("No. of DDSM Images:", normal_patches.shape)

#select 51.3% of DDSM data considering the number of CBIS data size

DDSM_norm_patches, X_norm_patches, DDSM_norm_Lbls, y_norm_Lbls, DDSM_norm_FNs, Z2_norm_FNs = \
train_test_split(normal_patches, normal_labels, normal_FNs, test_size = 0.487, random_state=20200121)


X_norm_train, X_norm_test, y_norm_train, y_norm_test, norm_FNs_train, norm_FNs_test = \
train_test_split(DDSM_norm_patches, DDSM_norm_Lbls, DDSM_norm_FNs, test_size = 0.183565, random_state=6325)

del DDSM_norm_patches
del DDSM_norm_Lbls
del DDSM_norm_FNs
gc.collect()

print("DDSM Train Images:", X_norm_train.shape)
print("DDSM Train Labels:", y_norm_train.shape)
print("DDSM Train File Names:", norm_FNs_train.shape)
print("\n")
print("DDSM Test Images:", X_norm_test.shape)
print("DDSM Test Labels:", y_norm_test.shape)
print("DDSM Test File Names:", norm_FNs_test.shape)

#check % of train data in the CBIS data and apply to DDSM train and test data split
pct_train = train_patches.shape[0]/(train_patches.shape[0]+test_patches.shape[0])
num_train_ddsm = normal_patches.shape[0]*pct_train
num_test_ddsm = normal_patches.shape[0]*(1-pct_train)

del normal_patches
del normal_labels
del normal_FNs
gc.collect()

print("% of preferred DDSM train data:", np.round(pct_train, 2))
print("Preferred No. of DDSM train data:", np.round(num_train_ddsm))
print("Preferred No. of DDSM test data:", np.round(num_test_ddsm))

"""**Merged train and test dataset**"""

#train data
train_images = np.concatenate([X_norm_train, train_patches], axis=0)
del X_norm_train
del train_patches
gc.collect()
train_labels = np.concatenate([y_norm_train, train_labels], axis=0)
del y_norm_train
gc.collect()
train_FNs = np.concatenate([norm_FNs_train, train_FNs], axis=0)
del norm_FNs_train
gc.collect()

#test data
test_images = np.concatenate([X_norm_test, test_patches], axis=0)
del X_norm_test
del test_patches
gc.collect()
test_labels = np.concatenate([y_norm_test, test_labels], axis=0)
del y_norm_test
gc.collect()
test_FNs = np.concatenate([norm_FNs_test, test_FNs], axis=0)
del norm_FNs_test
gc.collect()

"""**Label encoding**"""

le = preprocessing.LabelEncoder()
le.fit(train_labels)

list(le.classes_)

#Convert Normal to 0 
train_labels_en = le.transform(train_labels) + 1
train_labels_en[train_labels_en==5]=0

test_labels_en = le.transform(test_labels) + 1
test_labels_en[test_labels_en==5]=0

np.unique(train_labels_en)

np.unique(test_labels_en)

classes = le.classes_
classes = np.insert(classes, 0, 'NORMAL', axis=0)
classes = classes[0:5]

classes

train_bin_labels = np.zeros(len(train_labels_en)).astype(np.int32)
train_bin_labels[train_labels_en != 0] = 1

test_bin_labels = np.zeros(len(test_labels_en)).astype(np.int32)
test_bin_labels[test_labels_en != 0] = 1

np.unique(train_labels_en)

np.unique(train_bin_labels)

np.unique(test_labels_en)

np.unique(test_bin_labels)

"""**Save Labels**"""

np.save(os.path.join("Label", "train_labels_en.npy"), train_labels_en)
np.save(os.path.join("Label", "test_labels_en.npy"), test_labels_en)
np.save(os.path.join("Label", "train_bin_labels.npy"), train_bin_labels)
np.save(os.path.join("Label", "test_bin_labels.npy"), test_bin_labels)

"""**Distribution of data**"""

pd.value_counts(train_labels_en, normalize = True)

pd.value_counts(test_labels_en, normalize = True)

pd.value_counts(train_bin_labels, normalize = True)

pd.value_counts(test_bin_labels, normalize = True)

"""**Test and Validation Data Preparation**"""

X_val, X_test, y_val, y_test, y_val_multi, y_test_multi = \
    train_test_split(test_images, test_bin_labels, test_labels_en, test_size=0.5, random_state=19730104)
del test_images
del test_bin_labels
del test_labels_en
gc.collect()
X_train, y_train, y_train_multi = \
     shuffle(train_images, train_bin_labels, train_labels_en, random_state=100)
del train_images
del train_bin_labels
del train_labels_en
gc.collect()

X_train.shape

y_train.shape

"""**Save Final Data**"""

np.save(os.path.join("Data/256", 'X_train.npy'), X_train)
np.save(os.path.join("Data/256", 'y_train.npy'), y_train)
np.save(os.path.join("Data/256", 'train_labels_multi.npy'), y_train_multi)

np.save(os.path.join("Data/256", 'X_val.npy'), X_val)
np.save(os.path.join("Data/256", 'y_val.npy'), y_val)
np.save(os.path.join("Data/256", 'y_val_labels_multi.npy'), y_val_multi)

np.save(os.path.join("Data/256", 'X_test.npy'), X_test)
np.save(os.path.join("Data/256", 'y_test.npy'), y_test)
np.save(os.path.join("Data/256", 'y_test_labels_multi.npy'), y_test_multi)

