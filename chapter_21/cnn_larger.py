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
from keras.constraints import max_norm
from keras.optimizers import SGD

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

# # Plot ad hoc CIFAR10 instances
# for i in range(0, 9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(toimage(X_train[i]))
# plt.show()

# create baseline
epochs = 25
lrate = 0.01
decay = lrate / epochs


def baseline_model():
    # Create model
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_constraint=max_norm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_constraint=max_norm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    sgd = SGD(lr=lrate, momentum=.9, decay=decay, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = baseline_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=epochs)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1] * 100))
