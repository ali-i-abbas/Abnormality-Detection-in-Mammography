
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
# %matplotlib inline

#identify GPU
device_name = tf.test.gpu_device_name()
if not tf.test.is_gpu_available():
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.75)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))

print("TensorFlow version: ", tf.__version__)

try:
    os.makedirs("pretrained_model")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

"""**Loading dataset**"""

#train data
X_train = np.load(os.path.join("Data/256", "X_train.npy"))
y_train_bi = np.load(os.path.join("Data/256", "y_train.npy"))

#test data
X_test = np.load(os.path.join("Data/256", "X_test.npy"))
y_test_bi = np.load(os.path.join("Data/256", "y_test.npy"))

#validation data
X_val = np.load(os.path.join("Data/256", "X_val.npy"))
y_val_bi = np.load(os.path.join("Data/256", "y_val.npy"))

#train data
print("X_train data:", X_train.shape)
print("y_train data:", y_train_bi.shape)

#validation data
print("X_validation data:", X_val.shape)
print("y_validation data:", y_val_bi.shape)

#test data
print("X_test data:", X_test.shape)
print("y_test data:", y_test_bi.shape)

"""**Convert Label for categorical_crossentropy loss** """

#binary
y_train_bi = to_categorical(y_train_bi)
y_val_bi = to_categorical(y_val_bi)
y_test_bi = to_categorical(y_test_bi)

"""**Normalization**"""

# scale pixels
X_train = X_train/255.0
X_val = X_val/255.0
X_test = X_test/255.0

"""## Binary Classification CNN Model No. 5.2 - 4 VGG Blocks with BatchNormalization, Droput, and Weight Decay"""

classes = 2
def define_model21():
    model1 = Sequential()
    model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X_train.shape[1:]))
    model1.add(BatchNormalization())
    model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(BatchNormalization())
    model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Dropout(0.3))
    model1.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(BatchNormalization())
    model1.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Dropout(0.4))
    model1.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(BatchNormalization())
    model1.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Dropout(0.4))
    model1.add(Flatten())
    model1.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model1.add(BatchNormalization())
    model1.add(Dropout(0.5))
    model1.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model1.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model1

model21 = define_model21()

model21.summary()

# Change batch_size to 32 if you have a strong hardware
history21 = model21.fit(X_train, y_train_bi, epochs=50, batch_size=16, validation_data=(X_val, y_val_bi))

# save model and architecture to single file
model21.save("pretrained_model/ConvNet_binary2.h5")

_, acc = model21.evaluate(X_test, y_test_bi, verbose=0)
print('> %.3f' % (acc * 100.0))


accuracy = history21.history['accuracy']
val_accuracy = history21.history['val_accuracy']
loss = history21.history['loss']
val_loss = history21.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'ro', label='Train accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Train and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'ro', label='Train loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train and validation loss')
plt.legend()

plt.show()