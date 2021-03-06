'''
Using Dropout on the Visible Layer
'''

# Baseline Model on the Sonar Dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from pandas import read_csv
from keras.optimizers import SGD
from keras.constraints import maxnorm

seed = 7
np.random.seed(seed)

dataset = read_csv('./dataset/sonar.csv', delimiter=',', header=None)
X = dataset.iloc[:, :-1].values.astype(float)
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder

lb_encoder = LabelEncoder()
encoded_Y = lb_encoder.fit_transform(Y)


# baseline
def create_baseline():
    '''
    Dropout layer between the input (or visible layer) and the rst hidden layer.
    จะใช้ดรอปเอ้าท์ที่เลเยอร์ไหน ให้วางไว้ก่อนขึ้นโค้ดเลเยอร์นั้น
    :return:
    '''
    model = Sequential()

    model.add(Dropout(rate=.2, input_shape=(60,)))

    # maxnorm is set for control each weight in layer to can't over than 3,
    # because it maybe has some neuron have never in dropout set so it always update every cycle
    model.add(Dense(60, activation='relu', kernel_initializer='normal', kernel_constraint=maxnorm(3)))

    model.add(Dense(30, activation='relu', kernel_initializer='normal', kernel_constraint=maxnorm(3)))

    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))

    sgd = SGD(lr=.01, momentum=.9, decay=.0, nesterov=False)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    return model


estimators = []

from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier

estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=150, batch_size=16, verbose=0)))

from sklearn.pipeline import Pipeline

pipeline = Pipeline(estimators)

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

from sklearn.model_selection import cross_val_score

results = cross_val_score(estimator=pipeline, X=X, y=encoded_Y, cv=kfold)

print('Baseline: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))
