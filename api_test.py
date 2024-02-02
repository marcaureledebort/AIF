from flask import Flask, request, jsonify , send_file
from annoy import AnnoyIndex
import pandas as pd
from functions_aux import reco_imdb,reco_plot
import numpy as np
import io
from PIL import Image 
from sklearn.metrics.pairwise import cosine_distances

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


@app.route('/infos')
def infos():
    genres = metadata.columns[7:-1].tolist()  # Convertir l'Index en liste
    years = np.sort(metadata['year'].unique()).tolist()  # Convertir l'Index en liste
    return jsonify({'genres': genres, 'years': years})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #(debug=True) #