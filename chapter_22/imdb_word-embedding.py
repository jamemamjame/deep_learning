'''
Predict Sentiment From Movie Reviews
Word Embedding
'''

import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding

seed = 7
np.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# cut the length of word in each sentiment which longer than max_words, if lower will add 0 at front
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create model
def create_model():
    model = Sequential()
    # input_dim=number of distinct word in the dataset,
    # output_dim=size of the embedding vectors,
    # input_length=size of each input sequence
    model.add(Embedding(input_dim=top_words, output_dim=32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)

# Final evaluate of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1] * 100))
