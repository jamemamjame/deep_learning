'''
Visualize Model Training History

@author: Jame Phankosol
'''

# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("./dataset/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./chapter_15/_summary')
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0, callbacks=[tensorboard])

# list all data in history
history.history.keys()

# summarize history for accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])  # train
plt.plot(history.history['val_acc'])  # test
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
