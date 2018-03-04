'''
Predict Sentiment From Movie Reviews
Dataset Review
'''

import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot as plt

# load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# summarize size
print("Training data: ")
print(X.shape)
print(y.shape)
# Summarize number of classes
print("Classes: ")
print(np.unique(y))
# Summarize number of words
print("Number of words: ")
print(len(np.unique(np.hstack(X)))) # hstack is used to split nD array to 1D array
# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

# plot review length as a boxplot and histogram
plt.figure()
plt.subplot(121)
plt.ylabel('num word')
plt.boxplot(result)

plt.subplot(122)
plt.ylabel('freq')
plt.xlabel('num word/ document')
plt.hist(result)
plt.show()
