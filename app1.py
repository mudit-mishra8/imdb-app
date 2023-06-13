#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json
from gensim.models import Word2Vec
from keras.models import load_model


app = Flask(__name__)

# Load the model
LSTM_model = load_model('my_model.h9')
model = Word2Vec.load("word2vec.model")

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
                    if word in model.wv:
                        vectorized_review.append(model.wv[word])
                    else:
                        vectorized_review.append(np.zeros(model.wv.vector_size))  
                else:
                    vectorized_review.append(np.zeros(model.wv.vector_size)) 
            vectorized_data.append(vectorized_review)
        return vectorized_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])


def predict(): 
    rev = request.form['input_name']
    reviews = []
    reviews.append(rev.lower())
    sequences_validate = tokenizer.texts_to_sequences(reviews)
    # Define a function to replace words with vectors
    

    X_test_vectorized = replace_with_vectors(sequences_validate)

    # Pad sequences after vectorization
    max_length=200

    X_validate_padded = pad_sequences(X_test_vectorized, dtype='float32', padding='post', maxlen=max_length)
   

    #predictions = LSTM_model.predict(X_validate_padded)
    #if predictions[0][0] >=0.5:
    #    prediction= "Positive Review"
    #else:
      #  prediction= "Negative Review"

    return render_template('index.html', prediction_text= rev)


if __name__ == "__main__":
    app.run(debug=True)

