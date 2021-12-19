""" Contient les fonctions utilitaires de l'application. """

import pandas as pd
import numpy as np
import datetime
import csv
import os
from dateutil.parser import parse

def serie_generator():
    """ Génère un tirage aléatoire 
    
    Returns:
        list of int : une liste contenant les 5 numéros et les 2 numéros étoiles tirés
    """
    # Génération d'une suite de numéros
    boules = list(range(1, 51))
    numbers = np.random.choice(boules, 5, replace=False)
    stars = np.random.choice(boules[:12], 2, replace=False)

    liste = np.zeros(7)
    liste[:5] = numbers
    liste[5:7] = stars
    return(liste)

def lire_data_model():
    """ Lit les données du modèle dans le fichier dédié 
    
    Returns :
        le dictionnaire contenant toutes les données lues
    """
    data_model = pd.read_csv("app/static/data_model.csv", index_col=0)
    return data_model.to_dict()

def check_N(n):
    """ Vérifie un des numéros principaux 
    
    Params : 
        n (int): le numéro entré pour un des 5 numéros du tirage
    
    Returns :
        boolean : True si le numéro est valable (entre 1 et 50)
            False sinon
    """
    return (n<=50 and n>0)

def check_E(e):
    """ Vérifie qu'un numéro étoile est valable 
    
    Params : 
        e (int): le numéro entré pour un numéro étoile du tirage
    
    Returns :
        boolean : True si le numéro est valable (entre 1 et 12)
            False sinon
    """
    return (e<=12 and e>0)


def check_data(tirage):
    """ Vérifie que les données ont le bon format 

    Contraintes : 
        Les 5 numéros principaux doivent tous être entre 1 et 50
        Il ne doit pas y avoir de doublons dans les 5 numéros principaux 
        Les 2 numéros étoiles doivent être entre 1 et 12
        Les 2 numéros étoiles doivent être différents
    
    Params :
        tirage (Tirage): objet contenant les 5 numéros principaux et les 2 numéros étoiles
    
    Returns : 
        Boolean : True si le tirage est valable, False sinon
    """
    liste_N = [tirage.N1, tirage.N2, tirage.N3, tirage.N4, tirage.N5]
    nodoublons = (len(liste_N) == len(set(liste_N))) # False s'il y a des doublons
    N_okay = (nodoublons and check_N(tirage.N1) and check_N(tirage.N2) and check_N(tirage.N3) and check_N(tirage.N4) and check_N(tirage.N5))
    E_okay = ((tirage.E1 != tirage.E2) and check_E(tirage.E1) and check_E(tirage.E2))
    return (N_okay and E_okay)


def add_data(tirage, date, winner, gain):
    """ Ajoute une donnée dans le fichier de données 
    
    Params:
        tirage (Tirage): objet contenant les 5 numéros principaux et les 2 numéros étoiles d'un tirage
        date (datetime.date): la date du tirage
        winner (int): le nombre de gagnants du tirage
        gain (int): le gain du tirage, 0 si ce n'est pas un tirage gagnant
    
    Returns:
        None
    """
    file = "app/data/data.csv"
    rang = os.popen("wc -l "+file).read() # nombre de lignes dans le fichier
    rang = rang.split(" ")[0]
    ligne = [rang, date.strftime("%Y-%m-%d"), tirage.N1, tirage.N2, tirage.N3, tirage.N4, 
            tirage.N5, tirage.E1, tirage.E2, winner, gain]

    with open(file, 'a', newline='') as data:
        writerData  = csv.writer(data, delimiter=",")
        writerData.writerow(ligne)
    data.close()

    return None