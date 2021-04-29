import numpy as np
import pandas as pd
import os
from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import json
import sys
import shutil

dataframe = pd.read_csv('./Data/reviews_dataset.csv')


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


LSTM_ready_input_text_doc = create_OHE_padded_docs(dataframe['review_text'], 100000, 250)

vocab_size = 100000
max_length = 250
batch_size = 25
num_epochs = 7
num_features = 50

lstm_model = build_LSTM_model(vocab_size, max_length, n_features=num_features)

lstm_model.fit(LSTM_ready_input_text_doc, np.array(dataframe['label'].values), batch_size=batch_size,
               epochs=num_epochs)
'''
initial attempt to save the model--failed

filename = 'lstm_imdb_moview_model'
def save_model(model, filename):
    keras.models.save_model(model, filename)
save_model(lstm_model, filename)
'''


def save_model(model, absolute_filepath):
    try:
        os.mkdir(absolute_filepath)
    except Exception as e:
        print(e)
        resp = input("Do you want us remove the previous file and create new(y/n): ")
        if resp == 'y':
            shutil.rmtree(absolute_filepath)
            os.mkdir(absolute_filepath)
        else:
            print('Not saving the model')
            sys.exit()
    weights_path = os.path.join(absolute_filepath, 'weights')
    os.mkdir(weights_path)
    json_config = model.get_config()
    with open(os.path.join(absolute_filepath, 'model_config.txt'), 'w') as wrtfile:
        json.dump(json_config, wrtfile)
    model.save_weights(weights_path + "/")


save_model(lstm_model, './lstm_model')
