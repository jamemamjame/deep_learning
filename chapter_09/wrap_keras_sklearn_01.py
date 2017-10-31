'''
ฝึกใช้ KerasClassifier
cross_val_score ใช้ในการช่วยประเมินคุณภาพ model
'''

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

# Function to create model, required for KerasClassifier
def create_model():
    model = Sequential()
    model.add(Dense(units=12, input_dim=8, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt("dataset/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:-1]    # n_col=8
Y = dataset[:, -1]  # n_col=1

# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
result = cross_val_score(estimator=model, X=X, y=Y, cv=kfold)

print(result.mean())