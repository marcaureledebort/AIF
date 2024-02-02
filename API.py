#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:34:46 2024

@author: dumercq
"""

from flask import Flask, jsonify, request
from annoy import AnnoyIndex
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import re
import pickle
import numpy as np

with open('model.pk','rb') as F : 
    model=pickle.load(F)
    
nltk.download('punkt')
nltk.download('stopwords')
# Download stopwords list

stop_words = set(stopwords.words('english'))

# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]

tokenizer=StemTokenizer()

token_stop = tokenizer(' '.join(stop_words))

with open('Vectorizer.pk','rb') as F : 
    tfidf=pickle.load(F)
    
app = Flask(__name__)

db_annoy = AnnoyIndex(len(tfidf.get_feature_names_out()),metric='angular')
db_annoy.load("BOW_Embeddings.ann")

db_annoy_glove = AnnoyIndex(100,metric='angular')
db_annoy_glove.load("Glove_Embeddings.ann")


app = Flask(__name__)

words_list = model.index_to_key

@app.route('/text_reco', methods=['POST'])
def text_reco():
    text = request.json['text']
    embed = tfidf.transform([text])
    indices = db_annoy.get_nns_by_vector(embed.toarray()[0],n=5)
    return jsonify(indices=indices)

@app.route('/text_reco_glove', methods=['POST'])
def text_reco_glove():
    
    text = request.json['text']
    tok=tokenizer(text)
    embed = np.mean([model[w] for w in tok if w in words_list],axis=0)
    indices = db_annoy_glove.get_nns_by_vector(embed,n=5)
    return jsonify(indices=indices)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 

