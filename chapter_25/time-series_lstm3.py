'''
LSTM for Regression with Time Steps
'''

# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard


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


def create_dataset(dataset, look_back=1):
    '''
    convert an array of values into a dataset matrix
    :param dataset:
    :param look_back:
    :return:
    '''
    dataX, dataY = [], []
    for i in range(0, len(dataset) - 1 - look_back):
        dataX.append(dataset[i: i + look_back, 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def create_model():
    model = Sequential()

    model.add(LSTM(units=4, input_shape=(look_back, 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


seed = 7
np.random.seed(seed)

dataset = read_csv('./dataset/international-airline-passengers.csv', engine='python', delimiter=',', header=0,
                   usecols=[1], skip_footer=3)
X = dataset.values.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# split into train and test sets
trainset, testset = split_train_test(X, testsize=0.33)

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(trainset, look_back)
testX, testY = create_dataset(testset, look_back)

# reshape input to be [samples, time steps, features]
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

model = create_model()
tsboard = TensorBoard(log_dir='./chapter_25/_summary/lstm3')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[tsboard])

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(y_true=trainY[0], y_pred=trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train prediction for plotting
trainPredictPlot = np.empty_like(X)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# shift test prediction for plotting
testPredictPlot = np.empty_like(X)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + 1 + (2 * look_back): len(X) - 1, :] = testPredict

# plot baseline and predictions
plt.figure()
plt.grid(True)
plt.plot(scaler.inverse_transform(X))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# construct model
modelpath = './chapter_25/model_lstm3.json'
weightpath = './chapter_07/weight_lstm3.h5'
model_json = model.to_json()
with open(modelpath, 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weightpath)
print('Saved model to disk')

# later...
# load json and create model
from keras.models import model_from_json, model_from_yaml

json_file = open(modelpath, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(weightpath)
print("Loaded model from disk")
