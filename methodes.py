#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:24:25 2024

@author: dumercq
"""

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import re
from annoy import AnnoyIndex
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec

#--------------------------commun aux deux méthodes------------------------------
data = pd.read_csv("/home/dumercq/5GMM/AIF2024/projet/movies_metadata.csv")
data=data.dropna(subset=['overview'])
data=data[['overview', 'original_title']]
with open('Data.pk','wb')as f : 
    pickle.dump(data,f)
    
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

#---------------------------Bag of word------------------------------------------
tfidf = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer, max_features=400)
tfidf_matrix = tfidf.fit_transform(data.overview)
print(np.shape(tfidf_matrix))
with open('Vectorizer.pk','wb') as fin :
    pickle.dump(tfidf,fin)
    


dim = np.shape(tfidf_matrix)[1]

annoy_index = AnnoyIndex(dim,'angular')

features_list=[]

for i in tqdm(range(np.shape(tfidf_matrix)[0])) : 
    features_list.append(tfidf_matrix.getrow(i).toarray()[0])
    
for i, embedding in tqdm(enumerate(features_list)) : 
    annoy_index.add_item(i,embedding)

annoy_index.build(100)

annoy_index.save("BOW_Embeddings.ann")

#-----------------------------Glove----------------------------------------------
glove_file = ('glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
with open('model.pk','wb') as fin :
    pickle.dump(model,fin)
    
data['tokenized_overview'] = data['overview'].apply(lambda x: tokenizer(x))

bis=[]
features_list=[]
words_list = model.index_to_key

for i,word in enumerate (data.tokenized_overview) : 
    bis.append([model[w] for w in word if w in words_list])
    if bis[i]:
        features_list.append(np.mean(bis[i], axis=0))
    else:
        features_list.append(np.zeros(model.vector_size))

dim =100

annoy_index = AnnoyIndex(dim,'angular')

for i, embedding in tqdm(enumerate(features_list)) : 
    annoy_index.add_item(i,embedding)

annoy_index.build(100)

annoy_index.save("Glove_Embeddings.ann")