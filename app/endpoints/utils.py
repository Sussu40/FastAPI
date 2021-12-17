import pandas as pd
import numpy as np

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