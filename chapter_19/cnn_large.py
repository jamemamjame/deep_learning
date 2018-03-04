'''
Handwritten Digit Recognition
ิCNN Large

@author: jame phankosol
'''
# Simple CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt

# fix dimension ordering issue
from keras import backend as K

K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define a simple CNN model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(filters=30, kernel_size=(5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=15, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plotpic(digit_img):
    plt.subplot(111)
    plt.imshow(digit_img, cmap=plt.get_cmap('gray'))
    plt.show()


# build the model
model = baseline_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

# construct model
model_json = model.to_json()
with open('./chapter_19/large_model_structure.json', 'w') as f:
    f.write(model_json)
model.save_weights('./chapter_19/large_model_weight.h5')
print('Saved model to disk')

# load model
from keras.models import model_from_json

with open('./chapter_19/large_model_structure.json', 'r') as f:
    loaded_structure = f.read()
mymodel = model_from_json(loaded_structure)
mymodel.load_weights('./chapter_19/large_model_weight.h5')
print("Loaded model from disk")
