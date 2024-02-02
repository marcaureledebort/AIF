from flask import Flask, request, jsonify
import requests
import gradio as gr
import io
from PIL import Image 
import functools
import pickle
from io import BytesIO 

def reco_webapp(genre,year):
    rep=requests.post('http://127.0.0.1:5000/reco',json={'genre':genre,'year':year})

    data=rep.json()

    if data['title']=={}:
        msg="Aucun film trouvé sur la période et le genre spécifiés"
    else :
        msg="Voici les films les plus populaires de genre "+genre+" sortis en "+str(year)+"\n"
        films_tries = sorted(zip(data['title'].values(), data['imdb_score'].values()), key=lambda x: x[1], reverse=True)

        # Afficher les résultats
        for title, score in films_tries:
            msg+="\n "+ f"Film : {title}, Score IMDb : {round(score,2)}"
    return msg

def reco_plot_webapp(title):
    rep=requests.post('http://127.0.0.1:5000/reco_plot',json={'title':title})
    #rep=requests.post('http://annoy-db:5000/reco_plot',json={'title':title})
    if rep.status_code==200 :
        data=rep.json()   
        msg="Voici les films les plus proches de "+title +" en fonction de son plot \n"
        for title in data['title'].values():
            msg+="\n "+ f"- {title}"
    else: 
        msg="Ce film n'existe pas dans la base de données"
    return msg

def reco_poster_webapp(numero):
    response = requests.post("http://127.0.0.1:5000/reco_poster", json={"numero":numero}).json()
    #response = requests.post("http://annoy-db:5000/reco_poster", json={"numero":numero}).json()
    return [Image.open(i) for i in response['paths']]

def reco_poster_webapp_bis(image):
    image_bytes = BytesIO()
    image_pil = Image.fromarray(image)
    image_pil.save(image_bytes, format="JPEG")
    files = {'image': ('image.jpg', image_bytes.getvalue())}
    response = requests.post("http://127.0.0.1:5000/reco_poster_bis", files=files).json
    #response = requests.post("http://annoy-db:5000/reco_poster", json={"numero":numero}).json()
    return [Image.open(i) for i in response['paths']]




with open('Data.pk','rb') as f : 
    data_2=pickle.load(f)
    
def process_text(text,method,number_neighbor):
    #On choisit la méthode
    if method=="Bag of Words":
        response=requests.post('http://127.0.0.1:5000/text_reco',json={'text': text})
    else:
        response=requests.post("http://127.0.0.1:5000/text_reco_glove",json={'text': text})
    #Maintenant que l'on a choisi la méthode, on regarde si le statut code renvoie une erreur ou pas
    if response.status_code == 200:
        indices=response.json()['indices']
        titles=data_2.original_title.iloc[indices]
        overviews=data_2.overview.iloc[indices]

        return titles.tolist()[number_neighbor], overviews.tolist()[number_neighbor]
    else:
        return "Error in API request"   
    
with gr.Blocks() as demo:
    text=gr.Textbox(label='Write your movie description')
    method=gr.Radio(["Bag of Words","Glove"],label="Choose your method")
    btn = gr.Button(value="Submit")
        

    with gr.Accordion(label="Similar movies", open=True):
        for i in range(5):
            with gr.Row():
                title_output = gr.Textbox(label='Title')
                overview_output = gr.Textbox(label='Overview')
                btn.click(functools.partial(process_text,number_neighbor=i), inputs=[text, method], outputs=[title_output, overview_output])







if __name__=='__main__':
    response = requests.get('http://127.0.0.1:5000/infos')
    
    # Vérifiez si la requête a réussi (code d'état 200 OK)
    if response.status_code == 200:
        data = response.json()  # Désérialisez le contenu JSON
        genres = data.get('genres')  # Obtenez la valeur associée à la clé 'genres'
        years = data.get('years')    # Obtenez la valeur associée à la clé 'years'
        genres.insert(0,'Tous les genres')
        years.insert(0,'Toutes les années')

    i1=gr.Interface(fn=reco_webapp, 
                inputs=[gr.Dropdown(genres),
                        gr.Dropdown(years)
                        ], 
                outputs=['text'],
                description="entrez genre et année",
                )
    

    i2=gr.Interface(fn=reco_poster_webapp, 
                inputs="text", 
                outputs=['image','image','image','image','image'],
                description="Entrez un numéro de film entre 0 et 26937. La première image \
                    correspond au film selectionné, les 4 suivants aux films recommandés",
                )
    

    i3=gr.Interface(fn=reco_plot_webapp, 
                inputs=['text'], 
                outputs=['text'],
                description="Entrez un titre de film",
                )
    
    i4=gr.Interface(fn=reco_poster_webapp_bis, 
                inputs='image', 
                outputs=['image','image','image','image','image'],
                description="Entrez un titre de film",
                )
    


    gr.TabbedInterface([i1,i2,i3,i4,demo],["Recommandation par score IMdB","Recommandation par poster","Recommandation par plot","reco poster bis","Reco new"]).launch()