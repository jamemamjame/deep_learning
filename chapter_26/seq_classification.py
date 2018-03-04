'''
Project: Sequence Classification of Movie Reviews
'''

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

seed = 7
np.random.seed(seed)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_train, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32  # latent
model = Sequential()
# กำลังฝังตัวประโยคโดย embed output shape = จำนวนคำในประโยค * ลาเท้นของแต่ละคำ
model.add(Embedding(input_dim=top_words, output_dim=embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(units=100))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_test, y_test))

# Final evaluation of the model by the unseen data
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1] * 100))
