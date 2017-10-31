'''
Grid Search is use for find a best hyper_parameter
'''

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(units=12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=1, kernel_initializer=init, activation='sigmoid'))

    # compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('dataset/pima-indians-diabetes.csv', delimiter=',')

# split into input (X) and output (Y) variables
X = dataset[:, 0:-1]
Y = dataset[:, -1]

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
inits = ['glorot_uniform', 'uniform']
epochs = [50, 100, ]
batches = [5, 10]

param_grid = dict(  # name of key is required
    optimizer=optimizers,
    epochs=epochs,
    batch_size=batches,
    init=inits
)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)

# summarize results
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('mean: %f, stdev: %f, \tparam: %s' % (mean, stdev, param))
