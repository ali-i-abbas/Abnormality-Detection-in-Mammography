
# Commented out IPython magic to ensure Python compatibility.
import errno
import time
import numpy as np
import pandas as pd
import os
import sys
import sklearn
import datetime
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.backend import batch_normalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from packaging import version
import matplotlib.pyplot as plt
from numpy import loadtxt
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
# %matplotlib inline

#identify GPU
device_name = tf.test.gpu_device_name()
if not tf.test.is_gpu_available():
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.75)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))

print("TensorFlow version: ", tf.__version__)

"""## Load Saved Model"""


model3 = load_model('./pretrained_model/ConvNet_binary2.h5')

model3.summary()



#validation data
X_val = np.load(os.path.join("Data/256", "X_val.npy"))

#test data
X_test = np.load(os.path.join("Data/256", "X_test.npy"))

y_test_bi1 = y_test_bi = np.load(os.path.join("Data/256", "y_test.npy"))
y_val_bi1 = np.load(os.path.join("Data/256", "y_val.npy"))




"""## ROC AUC Curve - Binary Classification"""


def plot_roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)

def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize = [8,8])
    lw = 2
    plt.plot(fpr, tpr, color = 'darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle ='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show(block=False)
    plt.pause(0.001)


pred3 = model3.predict(X_val, batch_size = 32, verbose=0)
pred4 = np.argmax(pred3, axis=1)

pred7 = []
for i in range(pred3.shape[0]):
    pred7.append(np.min(pred3[i]))

pred8 = np.asarray(pred7)

pred8

fpr, tpr, thresholds = roc_curve(y_val_bi1, pred4)

plot_roc_curve(fpr, tpr)

fpr, tpr, thresholds = roc_curve(y_val_bi1, pred8)

plot_roc_curve(fpr, tpr)


plot_roc(pred8, y_val_bi1)

plt.show(block=False)
plt.pause(0.001)


"""## Confusion Matrix"""

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y1, pred_1)
from sklearn.metrics import auc
# auc_keras = auc(fpr_keras, tpr_keras)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Plot a confusion matrix.
# cm - confusion matrix
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


"""**Binary Class**"""

predictions_bi = model3.predict(X_test, verbose=0)
pred_bi = np.argmax(predictions_bi,axis=1)
instances_bi = pd.Index(['0','1'])

# Compute confusion matrix
cm = confusion_matrix(y_test_bi, pred_bi)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, instances_bi)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, instances_bi, title='Normalized confusion matrix')



"""## Precision, Recall, and F1 Score"""

from sklearn import metrics

"""**Binary Classisifcation**"""

print(metrics.confusion_matrix(y_test_bi, pred_bi))

print(metrics.classification_report(y_test_bi, pred_bi, digits=3))


plt.show()