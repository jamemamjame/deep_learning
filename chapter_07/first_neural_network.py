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
model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=150, batch_size=10, verbose=0)

# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# --------------------------------------------------------
# serialize model to JSON
'''
json save structure
HDF5 save weight
'''
model_json = model.to_json()
with open('./chapter_07/model.json', 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('./chapter_07/model.h5')
print('Saved model to disk')

# later...
# load json and create model
from keras.models import model_from_json, model_from_yaml

json_file = open('./chapter_07/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('./chapter_07/model.h5')
print("Loaded model from disk")
# --------------------------------------------------------

# ------------------------ Alternative for save model --------------------------------
# serialize model to YAML
model_yaml = model.to_yaml()
with open('./chapter_07/model.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

with open('./chapter_07/model.yaml', 'r') as yaml_file:
    loaded_model_yaml = yaml_file.read()
loaded_model = model_from_yaml(loaded_model_json)

# load weights into new model
loaded_model.load_weights('./chapter_07/model.h5')
print("Loaded model from disk")
# --------------------------------------------------------

# evaluate loaded model on test data
loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
