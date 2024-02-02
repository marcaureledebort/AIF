#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify , send_file
from annoy import AnnoyIndex
import pandas as pd
from functions_aux import reco_imdb,reco_plot
import numpy as np
import io
from PIL import Image 
from sklearn.metrics.pairwise import cosine_distances
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import re
import pickle
import torchvision.models as models
from io import BytesIO
import torch 
from torchvision import transforms
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
metadata=pd.read_csv('dataframe_imdb.csv')

indices = pd.read_csv('indices.csv', index_col='title').squeeze()
keywords = pd.read_csv("keywords.csv")
attributes_df = pd.read_csv('attributes.csv')
titles=pd.read_csv('titles.csv')
cosine_sim = cosine_distances(attributes_df.drop(columns=['id', 'title']))

model_path="./annoy_index.ann"
dim = 576
annoy_index = AnnoyIndex(dim, 'angular')
annoy_index.load(model_path)
features_df=pd.read_csv('features_df.csv',index_col=0)
features_df["features"] = features_df["features"].apply(lambda x: list(map(float, x.strip('[]').split())))



db_annoy = AnnoyIndex(len(tfidf.get_feature_names_out()),metric='angular')
db_annoy.load("BOW_Embeddings.ann")

db_annoy_glove = AnnoyIndex(100,metric='angular')
db_annoy_glove.load("Glove_Embeddings.ann")
words_list = model.index_to_key

@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    year = request.json['year'] # Get the vector from the request
    genre = request.json['genre'] # Get the vector from the request
    reco=reco_imdb(metadata,genre=genre,year=year).to_dict()
    return jsonify(reco) # Return the reco as a JSON

@app.route('/reco_plot', methods=['POST']) # This route is used to get recommendations
def reco_plot_api():
    title = request.json['title'] # Get the vector from the request
    reco=reco_plot(title,cosine_sim,indices,titles).to_dict()
    return jsonify(reco) # Return the reco as a JSON


@app.route('/reco_poster',methods=["POST"])
def reco_poster_api():
    idx = int(request.json['numero'])
    query_vector=features_df.features[idx]
    indices = annoy_index.get_nns_by_vector(query_vector, 5)
    paths = [features_df.path[idx] for idx in indices]
    return jsonify({'paths':paths})


@app.route('/reco_poster_bis',methods=["POST"])
def reco_poster_api_bis():
    image_file = request.files['image']
    image_pil = Image.open(BytesIO(image_file.read()))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    mobilenet = models.mobilenet_v3_small(pretrained=True)
    model_poster = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()).cuda()
    image_tensor = preprocess(image_pil).unsqueeze(0)
    model_poster.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_poster = model_poster.to(device)

    # Déplacer l'image_tensor sur le dispositif CUDA
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model_poster(image_tensor)
    output=output.view(-1)
    print(output.shape)
    indices = annoy_index.get_nns_by_vector(output, 5)
    paths = [features_df.path[idx] for idx in indices]
    return jsonify({'paths':paths})




@app.route('/infos')
def infos():
    genres = metadata.columns[7:-1].tolist()  # Convertir l'Index en liste
    years = np.sort(metadata['year'].unique()).tolist()  # Convertir l'Index en liste
    return jsonify({'genres': genres, 'years': years})





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
    app.run(debug=True)#(host='0.0.0.0', port=5000)  #