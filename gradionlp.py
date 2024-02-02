#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:04:15 2024

@author: dumercq
"""

import gradio as gr
import requests
import functools
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pickle
import os

with open('Data.pk','rb') as f : 
    data=pickle.load(f)
    
def process_text(text,method,number_neighbor):
    #On choisit la méthode
    if method=="Bag of Words":
        response=requests.post('http://127.0.0.1:5000/text_reco',json={'text': text})
    else:
        response=requests.post("http://127.0.0.1:5000/text_reco_glove",json={'text': text})
    #Maintenant que l'on a choisi la méthode, on regarde si le statut code renvoie une erreur ou pas
    if response.status_code == 200:
        indices=response.json()['indices']
        titles=data.original_title.iloc[indices]
        overviews=data.overview.iloc[indices]

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

if __name__ == "__main__":
    demo.launch()
