# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
from pandas import read_csv

seed = 7
np.random.seed(seed)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# load weights
model.load_weights(filepath='./chapter_07/weights.best.hdf5')

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# import dataset
dataset = read_csv('dataset/pima-indians-diabetes.csv', delimiter=',')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

