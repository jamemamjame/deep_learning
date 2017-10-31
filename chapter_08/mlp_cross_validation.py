'''
K-fold
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('dataset/pima-indians-diabetes.csv', delimiter=',')

X = dataset[:, 0:8]  # don't want column 8
Y = dataset[:, 8]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=1, shuffle=True, random_state=seed)

cvscores= []

for train, test in kfold.split(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(units=12, input_dim=8, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # fit model
    model.fit(X[train], Y[train], batch_size=10, epochs=150, verbose=0)

    # evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)

    cvscores.append(scores[1] * 100)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

