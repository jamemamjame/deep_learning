'''
Multiclass Classification
dataset: Iris Flowers Classification Dataset

@author: jame phankosol
'''

# Importing the library
import numpy as np
from astropy.table import np_utils
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
dataframe = read_csv('dataset/iris.csv', header=None)
dataset = dataframe.values
X = dataset[:, 0:-1].astype(float)
Y = dataset[:, -1]

# encode class values as integers
'''
convert A, B, C -> 0, 1, 2
'''
encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_Y = np_utils.to_categorical(encoder_Y)


# define baseline model
def baseline_model():
    '''
    4 inputs -> [8 hidden nodes] -> 3 outputs
    :return:
    '''
    # create a model
    model = Sequential()
    model.add(Dense(units=8, input_dim=4, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=False)

# Prepare Cross-Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

# evaluate our model
results = cross_val_score(estimator, X, dummy_Y, cv=kfold)

print('Accuracy = %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))
