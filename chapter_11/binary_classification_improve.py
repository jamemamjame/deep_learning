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

# load dataset
dataframe = read_csv('dataset/sonar.csv', header=None, delimiter=',')
dataframe = dataframe.values

# split into input (X) and output (Y) variables
X = dataframe[:, :-1].astype(float)
Y = dataframe[:, -1]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# baseline model
def create_baseline():
    '''
    Larger
    60 inputs -> [60] -> 1 output

    Larger
    60 inputs -> [60 -> 30] -> 1 output

    Smaller
    60 inputs -> [30] -> 1 output
    :return:
    '''
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# evaluate baseline model with standardized dataset
estimator = []
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(steps=estimator)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
