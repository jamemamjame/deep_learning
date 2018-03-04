'''
Boston Regression

@author: Jane Phankosol
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv

seed = 7
np.random.seed(seed)

dataframe = read_csv('./dataset/housing.csv', delimiter=',', header=0)
X = dataframe.iloc[:, :-1].values
Y = dataframe.iloc[:, -1].values


# define base model
def baseline_model():
    # create model
    model = Sequential()
    # --------------------------------------------------------------------
    # Deeper
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    #
    # Wider
    # model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    #
    # Normal
    # model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    # --------------------------------------------------------------------

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

model = KerasRegressor(baseline_model, epochs=50, batch_size=5, verbose=0)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', model))

from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=estimators)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(10, True, seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
