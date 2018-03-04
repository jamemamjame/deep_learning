'''
Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)

@author: jame phankosol
'''

# Import library
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def split_train_test(dataset, testsize=0.33):
    '''
    split data into train/test set
    :return:
    '''
    trainsize = 1. - testsize
    pivot = trainsize * len(dataset)
    X_train = dataset[:pivot, :]
    X_test = dataset[pivot:, :]
    return X_train, X_test


def create_dataset2(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def create_model(input_dim=1):
    model = Sequential()

    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(1))

    model.compile('adam', 'mean_squared_error')
    return model


seed = 7
np.random.seed(seed)

dataset = read_csv('./dataset/international-airline-passengers.csv', engine='python', delimiter=',', header=0,
                   usecols=[1], skip_footer=3)
X = dataset.values.astype('float32')

# split data into train and test set
trainset, testset = split_train_test(X, testsize=0.33)

# reshape data into X=t, Y=t+1
look_back = 1
trainX, trainY = create_dataset2(trainset, look_back=look_back)
testX, testY = create_dataset2(testset, look_back=look_back)

model = create_model(input_dim=look_back)
model.fit(x=trainX, y=trainY, batch_size=2, epochs=200, verbose=2)

# estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# generate prediction for training
trainPredict = model.predict(x=trainX)
testPredict = model.predict(x=testX)

# shift train prediction for plotting
trainPredictPlot = np.empty_like(X)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + 1, :] = trainPredict

# shift test prediction for plotting
testPredictPlot = np.empty_like(X)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + 1 + (2 * look_back): len(X) - 1, :] = testPredict

# plot graph
plt.figure()
plt.plot(X)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()