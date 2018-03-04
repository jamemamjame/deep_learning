from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.callbacks import ModelCheckpoint

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('dataset/pima-indians-diabetes.csv', delimiter=',')

X = dataset[:, 0:8]  # don't want column 8
Y = dataset[:, 8]

# create model
model = Sequential()
model.add(Dense(units=12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# checkpoint
'''
Checkpointing is set up to save the network weights only when there is an improvement in classification accuracy on the 
validation dataset (monitor=’val acc’ and mode=’max’).
'''
filepath = './chapter_07/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
filepath2 = './chapter_07/weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', mode='max', verbose=1, save_best_only=True)
checkpoint2 = ModelCheckpoint(filepath=filepath2, monitor='val_loss', mode='max', verbose=1, save_best_only=True)
callbacks_list = [checkpoint2]

# model.fit(x=X, y=Y, batch_size=10, epochs=150)
model.fit(X, Y, validation_split=.3, epochs=150, batch_size=10, verbose=0, callbacks=callbacks_list)

# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
