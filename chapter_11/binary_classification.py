'''
Multiclass Classification
dataset: Sonar Returns Classification Dataset

@author: jame phankosol
'''

import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

## เลือกสักวิธี
# load dataset (1)
dataframe = read_csv("dataset/sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:-1].astype(float)
Y = dataset[:, 60]

# load dataset as dataframe (2)
dataframe = read_csv('dataset/sonar.csv', header=None, delimiter=',')
# split into input and output variables
X = dataframe.iloc[:, 0:-1].values
Y = dataframe.iloc[:, -1].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(units=60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='normal', activation='sigmoid'))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

print('Accuracy = %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))
