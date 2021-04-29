from flask import (Flask, render_template, abort,url_for,
                   jsonify, request, redirect)
import re
import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import os
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

app=Flask(__name__)


def model_loader(modelpath):
    weights_path = modelpath + '/weights/'
    config_file = os.path.join(modelpath, 'model_config.txt')
    with open(config_file) as f:
        json_config = json.load(f)
    print(json_config)
    model_arch = tf.keras.Sequential.from_config(json_config)
    model_arch.summary()
    model_arch.load_weights(weights_path)

    return model_arch

model=model_loader('./lstm_model')

@app.route("/", methods=["GET","POST"])
def welcome():
    if request.method == "POST":
        review = request.form['review']
        review = prepare_review(review)
        print(review)
        input_ready_review = create_OHE_padded_docs([review], 100000, 250)
        print('sending for prediction')
        prediction = model.predict(input_ready_review)
        print(prediction)
        sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
        return render_template("welcome.html",result=sentiment)
    else:
        return render_template("welcome.html")


def create_OHE_padded_docs(text_doc, vocab_size, max_length):
    print('pre processing started')
    embedded_docs = [one_hot(i, n=vocab_size) for i in text_doc]
    padded_embedded_docs = pad_sequences(embedded_docs, maxlen=max_length, padding='post', truncating='post')
    print('preprocesing complted')
    return padded_embedded_docs


def prepare_review(review):
    only_words=re.findall('[a-z0-9]+',review,re.IGNORECASE)
    only_text=' '.join(only_words)
    return only_text


if __name__ =='__main__':
    app.run(debug=True)