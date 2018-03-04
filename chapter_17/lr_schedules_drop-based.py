'''
Drop-Based Learning Rate Schedule

must define a new self function for step a learning rate
                                            floor((1+Epoch)/EpochDrop)
LearningRate = InitialLearningRate x DropRate
'''

# import library
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
from keras.optimizers import SGD


def step_decay(epoch):
    initial_lrate = .1
    drop_rate = .5
    epochs_drop = 10.0

    # if n_epoch = [31, 40] >> lrate = init_lrate x (0.5 x 0.5 x 0.5) ; because 35/10 = 3
    lrate = initial_lrate * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
    return lrate

import scrapy

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
sgd = SGD(lr=.0, momentum=.9, decay=.0, nesterov=False)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# learning schedule callback
from keras.callbacks import LearningRateScheduler

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# Fit the model (train)
# verbose=2 is 1 line/epoch
model.fit(X, encoded_Y, batch_size=28, epochs=50, validation_split=.33, callbacks=callbacks_list, verbose=2)

