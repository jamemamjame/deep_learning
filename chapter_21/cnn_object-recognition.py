'''
Project Object Recognition in Photographs
Photographs from Canadian Institute for Advanced Research (CIFAR)

@author: jame phankosol
'''

# Importing the library
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import backend as K

K.set_image_dim_ordering('th')

# fix seed for random
seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_train.shape[1]

# hyper-parameter variable
epochs = 25
lrate = 0.01
decay = lrate / epochs


# create baseline

def baseline_model():
    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    sgd = SGD(lr=lrate, momentum=.9, decay=decay, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = baseline_model()

# check point
filepath = './chapter_21/_checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
log_dir = './chapter_21/_summary'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', mode='max', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir=log_dir)
callback_list = [checkpoint, tensorboard]

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=200, epochs=epochs, callbacks=callback_list)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1] * 100))
