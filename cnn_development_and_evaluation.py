
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
y_train = np.load(os.path.join("Data/256", "train_labels_multi.npy"))
y_train_bi = np.load(os.path.join("Data/256", "y_train.npy"))

#test data
X_test = np.load(os.path.join("Data/256", "X_test.npy"))
y_test = np.load(os.path.join("Data/256", "y_test_labels_multi.npy"))
y_test_bi = np.load(os.path.join("Data/256", "y_test.npy"))

#validation data
X_val = np.load(os.path.join("Data/256", "X_val.npy"))
y_val = np.load(os.path.join("Data/256", "y_val_labels_multi.npy"))
y_val_bi = np.load(os.path.join("Data/256", "y_val.npy"))

#train data
print("X_train data:", X_train.shape)
print("y_train data:", y_train.shape)

#validation data
print("X_validation data:", X_val.shape)
print("y_validation data:", y_val.shape)

#test data
print("X_test data:", X_test.shape)
print("y_test data:", y_test.shape)

"""**Convert Label for categorical_crossentropy loss** """

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#binary
y_train_bi = to_categorical(y_train_bi)
y_val_bi = to_categorical(y_val_bi)
y_test_bi = to_categorical(y_test_bi)

"""**Normalization**"""

# scale pixels
X_train = X_train/255.0
X_val = X_val/255.0
X_test = X_test/255.0

"""## CNN Model No. 1 with 1 VGG Block"""

classes = 5
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

# fit model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

_, acc = model.evaluate(X_test, y_test, verbose=0)

print('> %.3f' % (acc * 100.0))

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history.history['loss'], color='blue', label='train')
ax1.plot(history.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history.history['accuracy'], color='blue', label='train')
ax2.plot(history.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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

"""## CNN Model No. 2 with 2 VGG Blocks"""

classes = 5
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

_, acc = model.evaluate(X_test, y_test, verbose=0)

print('> %.3f' % (acc * 100.0))

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history.history['loss'], color='blue', label='train')
ax1.plot(history.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history.history['accuracy'], color='blue', label='train')
ax2.plot(history.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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

"""## CNN Model No. 3 with 3 VGG Blocks"""

classes = 5
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

_, acc = model.evaluate(X_test, y_test, verbose=0)

print('> %.3f' % (acc * 100.0))

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history.history['loss'], color='blue', label='train')
ax1.plot(history.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history.history['accuracy'], color='blue', label='train')
ax2.plot(history.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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

"""## CNN Model No. 4 3VGG Blocks with Dropout"""

classes = 5
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

_, acc = model.evaluate(X_test, y_test, verbose=0)

print('> %.3f' % (acc * 100.0))

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history.history['loss'], color='blue', label='train')
ax1.plot(history.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history.history['accuracy'], color='blue', label='train')
ax2.plot(history.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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

"""## CNN Model No. 5 3 VGG Blocks with Dropout variation and Batch Normalization"""

classes = 5
def define_model1():
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
    model1.add(Flatten())
    model1.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model1.add(BatchNormalization())
    model1.add(Dropout(0.5))
    model1.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model1

model1 = define_model1()

history1 = model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

_, acc = model1.evaluate(X_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history1.history['accuracy']
val_accuracy = history1.history['val_accuracy']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
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

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history1.history['loss'], color='blue', label='train')
ax1.plot(history1.history['val_loss'], color='orange', label='validation')
ax1.legend()

# plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history1.history['accuracy'], color='blue', label='train')
ax2.plot(history1.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# save model and architecture to single file
model1.save("pretrained_model/ConvNet_multiNo5_1.h5")

"""**Old Result**"""

history1 = model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

_, acc = model1.evaluate(X_test, y_test, verbose=0)

print('> %.3f' % (acc * 100.0))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history1.history['accuracy']
val_accuracy = history1.history['val_accuracy']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
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

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history1.history['loss'], color='blue', label='train')
ax1.plot(history1.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history1.history['accuracy'], color='blue', label='train')
ax2.plot(history1.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

"""## CNN Model No. 5.2 4 VGG Blocks with Dropout and BatchNormalization """

classes = 5
def define_model12():
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
    model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model1

model12 = define_model12()

history12 = model12.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# save model and architecture to single file
model12.save("pretrained_model/ConvNet_multiNo5_2.h5")

_, acc = model12.evaluate(X_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history12.history['loss'], color='blue', label='train')
ax1.plot(history12.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history12.history['accuracy'], color='blue', label='train')
ax2.plot(history12.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history12.history['accuracy']
val_accuracy = history12.history['val_accuracy']
loss = history12.history['loss']
val_loss = history12.history['val_loss']
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



classes = 5
def define_model13():
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
    model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model1

model13 = define_model13()

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

class_weights

history13 = model13.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), class_weight=class_weights)

# save model and architecture to single file
model13.save("pretrained_model/ConvNet_multiNo53_w.h5")

_, acc = model13.evaluate(X_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history13.history['accuracy']
val_accuracy = history13.history['val_accuracy']
loss = history13.history['loss']
val_loss = history13.history['val_loss']
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

"""## CNN Model No. 5.3 3 VGG Blocks with Dropout, BatchNormalization and Weight Decay"""

classes = 5
param = 0.01
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param), input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model2 = define_model()

history2 = model2.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

_, acc = model2.evaluate(X_test, y_test, verbose=0)

print('> %.3f' % (acc * 100.0))

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history2.history['loss'], color='blue', label='train')
ax1.plot(history2.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history2.history['accuracy'], color='blue', label='train')
ax2.plot(history2.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history2.history['accuracy']
val_accuracy = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
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

# save model and architecture to single file
model2.save("trained_model/ConvNet1.h5")

"""## Binary Classification CNN Model No. 5.1 - 3 VGG Blocks with BatchNormalization, Droput, and Weight Decay"""

classes = 2
param = 0.01
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param), input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(param)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model3 = define_model()

history3 = model3.fit(X_train, y_train_bi, epochs=50, batch_size=32, validation_data=(X_val, y_val_bi))

# save model and architecture to single file
model3.save("pretrained_model/ConvNet_binary1.h5")

_, acc = model3.evaluate(X_test, y_test_bi, verbose=0)

print('> %.3f' % (acc * 100.0))

ax1 = plt.subplot(211)
plt.title('Cross Entropy Loss')
ax1.plot(history3.history['loss'], color='blue', label='train')
ax1.plot(history3.history['val_loss'], color='orange', label='validation')
ax1.legend()
    # plot accuracy
ax2 = plt.subplot(212)
plt.title('Classification Accuracy')
ax2.plot(history3.history['accuracy'], color='blue', label='train')
ax2.plot(history3.history['val_accuracy'], color='orange', label='validation')
ax2.legend()
plt.tight_layout()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history3.history['accuracy']
val_accuracy = history3.history['val_accuracy']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
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

history21 = model21.fit(X_train, y_train_bi, epochs=50, batch_size=32, validation_data=(X_val, y_val_bi))

# save model and architecture to single file
model21.save("pretrained_model/ConvNet_binary2.h5")

_, acc = model21.evaluate(X_test, y_test_bi, verbose=0)
print('> %.3f' % (acc * 100.0))

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_bi), y_train_bi)

class_weights

history21_w = model21.fit(X_train, y_train_bi, epochs=50, batch_size=32, validation_data=(X_val, y_val_bi), class_weight=class_weights)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

accuracy = history21_w.history['accuracy']
val_accuracy = history21_w.history['val_accuracy']
loss = history21_w.history['loss']
val_loss = history21_w.history['val_loss']
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

# save model and architecture to single file
model21.save("pretrained_model/ConvNet_binary2_class_weight.h5")

"""## Load Saved Model"""

from numpy import loadtxt
from tensorflow.keras.models import load_model

model_multi_52 = load_model('./pretrained_model/ConvNet_multiNo5_2.h5')

model_multi_52.summary()

y_train1 = np.load(os.path.join("Data/256", "train_labels_multi.npy"))
y_train_bi1 = np.load(os.path.join("Data/256", "y_train.npy"))
y_test1 = np.load(os.path.join("Data/256", "y_test_labels_multi.npy"))
y_test_bi1 = y_test_bi = np.load(os.path.join("Data/256", "y_test.npy"))
y_val1 = np.load(os.path.join("Data/256", "y_val_labels_multi.npy"))
y_val_bi1 = np.load(os.path.join("Data/256", "y_val.npy"))

"""## ROC AUC Curve - Binary Classification"""

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

pred3 = model3.predict(X_val, batch_size = 32, verbose=0)
pred4 = np.argmax(pred3, axis=1)

pred7 = []
for i in range(4656):
    pred7.append(np.min(pred3[i]))

pred8 = np.asarray(pred7)

pred8

fpr, tpr, thresholds = roc_curve(y_val_bi1, pred4)

plot_roc_curve(fpr, tpr)

fpr, tpr, thresholds = roc_curve(y_val_bi1, pred8)

plot_roc_curve(fpr, tpr)

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
    plt.show()

plot_roc(pred8, y_val_bi1)

"""**CNN Model No. 5.2**"""

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

pred21 = model_multi_52.predict(X_test, batch_size = 32, verbose=0)
pred22 = np.argmax(pred21, axis=1)

pred9 = []
for i in range(4656):
    pred9.append(np.min(pred21[i]))
pred10 = np.asarray(pred9)

fpr, tpr, thresholds = roc_curve(y_test_bi1, pred10)

plot_roc_curve(fpr, tpr)

pred11 = []
for i in range(4656):
    pred11.append(np.max(pred21[i]))
pred12 = np.asarray(pred11)

fpr, tpr, thresholds = roc_curve(y_test_bi1, pred12)

plot_roc_curve(fpr, tpr)

"""## ROC AUC Curve - Multi Classification"""

from scipy import interpolate
from itertools import cycle
from sklearn.metrics import roc_curve, auc

pred_mul = model1.predict(X_val, batch_size = 32, verbose = 0)
#pred_mul1 = np.argmax(pred_mul, axis=1)

pred_mul

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_val.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], pred_mul[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), pred_mul.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interpolate(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize = [8, 8])
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'yellowgreen', 'silver', 'crimson'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle ='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize = [8, 8])
lw = 2
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'yellowgreen', 'silver', 'crimson'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle ='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

"""**CNN Model No. 5.2**"""

predictions5_2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred21[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred21.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1, figsize = [8, 8])
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'yellowgreen', 'silver', 'crimson'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle ='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# Zoom in view of the upper left corner.
# Plot all ROC curves
plt.figure(figsize = [8, 8])
lw = 2

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'yellowgreen', 'silver', 'crimson'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle ='--')
plt.xlim([0.0, 0.2])
plt.ylim([0.8, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

"""## Calculate Classification Log Loss"""

from IPython.display import display

# Don't display numpy in scientific notation
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Generate predictions
pred21

print("Numpy array of predictions")
display(pred21[0:5])

print("As percent probability")
print(pred21[0]*100)

score = metrics.log_loss(y_test, pred21)
print("Log loss score: {}".format(score))

pred = np.argmax(pred21,axis=1) # raw probabilities to chosen class (highest probability)

"""## Confusion Matrix"""

from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import roc_curve
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(y1, pred_1)
from sklearn.metrics import auc
#auc_keras = auc(fpr_keras, tpr_keras)

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
    

# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

instances = pd.Index(['0','1','2','3','4'])

instances1 = pd.Index(['1','2','3','4', '0'])

instances1

"""**Multiclass**"""

# Compute confusion matrix
y_compare = np.argmax(y_test, axis =1)
cm = confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, instances)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, instances, title='Normalized confusion matrix')

plt.show()

"""**CNN Model No. 5.2**"""

pred5_2 = model12.predict(X_test, verbose=0)

pred1 = np.argmax(pred5_2,axis=1)

# Compute confusion matrix
y_compare = np.argmax(y_test, axis =1)
cm = confusion_matrix(y_compare, pred22)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, instances)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, instances, title='Normalized confusion matrix')

plt.show()



"""**CNN Model No. 5.2_weight**"""

from sklearn.metrics import plot_confusion_matrix

pred13_w = model13.predict(X_test, verbose=0)

y_compare_13_w = np.argmax(pred13_w,axis=1)

from sklearn.metrics import plot_confusion_matrix

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, 
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

# Compute confusion matrix
y_compare = np.argmax(y_test, axis =1)
cm = confusion_matrix(y_compare, y_compare_13_w)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, instances)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, instances, title='Normalized confusion matrix')

plt.show()

# Compute confusion matrix
y_compare = np.argmax(y_test, axis =1)
cm = confusion_matrix(y_compare, y_compare_13_w)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, instances)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, instances, title='Normalized confusion matrix')

plt.show()



"""**Binary Class**"""

model_bi = load_model('./pretrained_model/ConvNet1.h5')
predictions_bi = model_bi.predict(X_test, verbose=0)
pred_bi = np.argmax(predictions_bi,axis=1)
instances_bi = pd.Index(['0','1'])

# Compute confusion matrix
y_compare_bi = np.argmax(y_test_bi, axis =1)
cm = confusion_matrix(y_compare_bi, pred_bi)
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

plt.show()

"""**CNN Model No. 5.2**"""

#predictions21 = model21.predict(X_test, verbose=0)
pred22 = np.argmax(pred21,axis=1)
instances_bi = pd.Index(['0','1'])

# Compute confusion matrix
y_compare_bi = np.argmax(y_test_bi, axis =1)
cm = confusion_matrix(y_compare_bi, pred21)
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

plt.show()



"""## Precision, Recall, and F1 Score"""

from sklearn import metrics

"""**Binary Classisifcation**"""

print(metrics.confusion_matrix(y_compare_bi, pred_bi))

print(metrics.classification_report(y_compare_bi, pred_bi, digits=3))

"""**CNN Model No. 5.2**"""

print(metrics.confusion_matrix(y_compare_bi, pred22))

print(metrics.classification_report(y_compare_bi, pred22, digits=3))

"""## Multiclass Classification"""

print(metrics.confusion_matrix(y_compare, pred))

print(metrics.classification_report(y_compare, pred, digits=3))

"""**CNN Model No. 5.2**"""

print(metrics.confusion_matrix(y_compare, pred1))

print(metrics.classification_report(y_compare, pred1, digits=3))



"""**CNN Model No. 5.3 class_weight**"""

print(metrics.confusion_matrix(y_compare, y_compare_13_w))

print(metrics.classification_report(y_compare, y_compare_13_w, digits=3))



"""## Precision Recall Curve"""

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

n_classes = y_test.shape[1]

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],predictions5_1[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], predictions5_1[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    predictions5_1.ravel())
average_precision["micro"] = average_precision_score(y_test, predictions5_1,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

"""**Micro-Averaged Precision-Recall Curve** """

plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))

"""**Precision-Recall Curve for each class and iso-f1 curves**"""

from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(8,9))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
plt.tight_layout()
plt.show()

"""**CNN Model No. 5.2**"""

n_classes = y_test.shape[1]

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],predictions5_2[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], predictions5_2[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    predictions5_2.ravel())
average_precision["micro"] = average_precision_score(y_test, predictions5_2,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))

from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(8,9))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
plt.tight_layout()
plt.show()

"""## Predictions"""

y_test1 = np.load(os.path.join("Data/256", "y_test_labels_multi.npy"))

class_names = ['NORMAL', 'BENIGN_calcification', 'BENIGN_mass',
       'MALIGNANT_calcification', 'MALIGNANT_mass']

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i].reshape(256,256)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(5))
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, pred21[i], y_test1, X_test)
plt.subplot(1,2,2)
plot_value_array(i, pred21[i],  y_test1)
plt.show()

predictions2 = model2.predict(X_test, verbose=0)

y_test1 = np.load(os.path.join("Data/256", "y_test_labels_multi.npy"))

class_names = ['NORMAL', 'BENIGN_calcification', 'BENIGN_mass',
       'MALIGNANT_calcification', 'MALIGNANT_mass']

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i].reshape(256,256)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% \n ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(5))
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

"""**CNN Model No.5**"""

list1 = [2, 32, 81, 82, 103, 89, 5, 90, 104, 93, 100, 136, 139, 140, 121]
num_rows = 30
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i,j in enumerate(list1):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(j, pred21[j], y_test1, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(j, pred21[j], y_test1)
plt.tight_layout()
plt.show()







import keras
import pydotplus
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

from tensorflow.keras.utils import plot_model

tf.keras.utils.plot_model(model_multi_52, to_file='model.png')

from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model2).create(prog='dot', format='svg'))

predictions2 = model1.predict(X_test, verbose=0)

num_rows = 30
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(0, 1500,1):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions2[i], y_test1, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions2[i], y_test1)
plt.tight_layout()
plt.show()

