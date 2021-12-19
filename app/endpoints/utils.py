import pandas as pd
import numpy as np
import datetime
import csv
import os

def serie_generator():
    """ Génère un tirage aléatoire """
    # Génération d'une suite de numéros
    boules = list(range(1, 51))
    numbers = np.random.choice(boules, 5, replace=False)
    stars = np.random.choice(boules[:12], 2, replace=False)

    liste = np.zeros(7)
    liste[:5] = numbers
    liste[5:7] = stars

    return(liste)

def lire_data_model():
    """ Lit les données du modele dans le fichier dédié """
    data_model = pd.read_csv("app/static/data_model.csv", index_col=0)
    return data_model.to_dict()

def add_data(tirage, date, winner, gain):
    """ Ajoute une donnee dans le fichier de donnees """
    file = "app/data/data.csv"
    rang = os.popen("wc -l "+file).read() # nombre de lignes dans le fichier
    rang = rang.split(" ")[0]
    print(rang)
    ligne = [rang, date.strftime("%Y-%m-%d"), tirage.N1, tirage.N2, tirage.N3, tirage.N4, tirage.N5, tirage.E1,
            tirage.E2, winner, gain]

    with open(file, 'a', newline='') as data:
        writerData  = csv.writer(data, delimiter=",")
        writerData.writerow(ligne)
    data.close()

    return None