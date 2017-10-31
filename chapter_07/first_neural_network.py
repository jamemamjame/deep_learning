from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('dataset/pima-indians-diabetes.csv', delimiter=',')

X = dataset[:, 0:8]  # don't want column 8
Y = dataset[:, 8]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.33, random_state=seed)

# create model
model = Sequential()
model.add(Dense(units=12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit model
'''
validation_set is used to reduce over-fitting
'''
# model.fit(x=X, y=Y, batch_size=10, epochs=150)
model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=150, batch_size=10,)

# evaluate the model
scores = model.evaluate(X, Y,)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

