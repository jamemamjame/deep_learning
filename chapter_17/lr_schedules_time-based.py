'''
Time-Based Learning Rate Schedule

in SGD: decay variable is used for..
LearningRate =  LearningRatex _________1__________
                                1 + decay x epoch
'''

# import library
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
from keras.optimizers import SGD

seed = 7
np.random.seed(seed)

dataset = read_csv("./dataset/ionosphere.csv", header=None)
X = dataset.iloc[:, :-1].values.astype(float)
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder

lb_encoder = LabelEncoder()
encoded_Y = lb_encoder.fit_transform(Y)

# create model
model = Sequential()
model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# compile model
learning_rate = 0.1
epochs = 50
decay = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model (train)
# verbose=2 is 1 line/epoch
model.fit(X, encoded_Y, batch_size=28, epochs=epochs, validation_split=.33, verbose=2)
