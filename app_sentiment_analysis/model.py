import numpy as np
import re
import os
from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import random

pos_train_path = 'C:/Users/91911/Documents/Python Scripts/Data/Imdb/aclImdb/train/pos/'
neg_train_path = 'C:/Users/91911/Documents/Python Scripts/Data/Imdb/aclImdb/train/neg/'
len(os.listdir(pos_train_path))

def get_data_with_label(path, positive=True):
    """set positive as False if you want to get negative reviews"""
    label = 1 if positive else 0
    reviews = []
    for data_file in os.listdir(path):
        with open(path + data_file, 'rb') as reader:
            only_words = re.findall('[a-z0-9]+', reader.readline().decode(), re.IGNORECASE)
            only_text = ' '.join(only_words)
            reviews.append((only_text, label))
    return reviews

positive_reviews = get_data_with_label(pos_train_path)

negative_reviews = get_data_with_label(neg_train_path, positive=False)

pos_and_neg_reviews = positive_reviews + negative_reviews

random.shuffle(pos_and_neg_reviews)
reviews = [i for i, j in pos_and_neg_reviews]

labels = [j for i, j in pos_and_neg_reviews]


def create_OHE_padded_docs(text_doc, vocab_size, max_length):
    embedded_docs = [one_hot(i, n=vocab_size) for i in text_doc]
    padded_embedded_docs = pad_sequences(embedded_docs, maxlen=max_length, padding='post', truncating='post')

    return padded_embedded_docs


def build_LSTM_model(vocab_size, max_length, n_features):
    model = Sequential()
    model.add(keras.layers.Embedding(vocab_size, n_features, input_length=max_length, mask_zero=True))
    model.add(keras.layers.LSTM(4, input_shape=[None, max_length, n_features]))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae'])

    model.summary()

    return model


LSTM_ready_input_text_doc = create_OHE_padded_docs(reviews, 100000, 250)

vocab_size = 100000
max_length = 250
batch_size = 25
num_epochs = 20
num_features = 50


lstm_model = build_LSTM_model(vocab_size, max_length, n_features=num_features)


lstm_model.fit(LSTM_ready_input_text_doc, np.array(labels), batch_size=batch_size,
               epochs=num_epochs)

filename = 'lstm_imdb_moview_model'


def save_model(model, filename):
    keras.models.save_model(model, filename)


save_model(lstm_model, filename)






