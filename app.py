#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
import gensim
from gensim.models import KeyedVectors

app = Flask(__name__)

# Load the model
LSTM_model = load_model('my_model.h5')
word_vectors = KeyedVectors.load("word_vectors.kv")

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def replace_with_vectors(data):
    vectorized_data = []
    for review in data:
        vectorized_review = []
        for word_index in review:
            if word_index in tokenizer.index_word:
                word = tokenizer.index_word[word_index]
                if word in word_vectors:
                    vectorized_review.append(word_vectors[word])
                else:
                    vectorized_review.append(np.zeros(word_vectors.vector_size))
            else:
                vectorized_review.append(np.zeros(word_vectors.vector_size))
        vectorized_data.append(vectorized_review)
    return vectorized_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    rev = request.form['input_name']
    reviews = [rev.lower()]
    sequences_validate = tokenizer.texts_to_sequences(reviews)

    X_test_vectorized = replace_with_vectors(sequences_validate)
    max_length = 200
    X_validate_padded = pad_sequences(X_test_vectorized, dtype='float32', padding='post', maxlen=max_length)

    predictions = LSTM_model.predict(X_validate_padded)
    if predictions[0][0] >= 0.5:
        prediction = "Positive Review"
    else:
        prediction = "Negative Review"

    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)
