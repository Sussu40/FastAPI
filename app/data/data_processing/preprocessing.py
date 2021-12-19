import pandas as pd
import numpy as np
from ...endpoints.utils import serie_generator

# Objectif du fichier : créer un nouveau csv contenant des séries perdantes et la variable cible


def data_generator(data, date):
    """ Génération d'une suite de numéros et l'ajoute aux données
    
    Params:
        data (pd.df): dataframe contenant la liste de tirages gagnants
        date (str): date du tirage généré

    Returns:
        pd.df : le dataframe avec le nouveau tirage ajouté
    """
    liste = list(np.zeros(9)) # ligne de données
    # liste[0] = date # premier élément = date du tirage
    liste[:7] = serie_generator() # 7 données suivantes = un tirage aléatoire
    # 2 dernières données = Winner, Gain (nulles car tirage perdant)
    
    # ajout au DF
    df = pd.DataFrame([[date]+liste], columns=list(data.columns), index=[str(len(data))])
    data = data.append(df)
    return data


def data_generator_x_sorted(data, file, x:int=100):
    """ Génère x tirages non-gagnants par tirage gagnant
        les ajoute au dataframe des données
        Puis enregistre le tout dans un fichier (csv)

    Params:
        data (pd.df): dataframe contenant la liste de tirages gagnants
        x (int): nombre de tirages perdants à ajouter à chaque tirage gagnant
        file (str): nom du fichier ou seront enregistrees les donnees

    Returns:
        None
    """
    size = len(data)
    columns = list(data.columns)
    # initialisation du df vide
    data_bis = pd.DataFrame(columns = columns)
    # on ajoute x données fausses pour une vraie
    for k in range(size):
        # trier les nombres du tirage
        N = list(sorted(data[["N1", "N2", "N3", "N4", "N5"]].iloc[k]))
        E = list(sorted(data[["E1", "E2"]].iloc[k]))
        date = data["Date"][k]
        wg = list(data[["Winner", "Gain"]].iloc[k])
        df = pd.DataFrame([[date]+N+E+wg], columns=columns, index=[str(len(data_bis))])
        # on ajoute le tirage gagnant au df
        data_bis = data_bis.append(df)
        # puis on ajoute x tirages perdants
        for i in range(x) : 
            data_bis = data_generator(data_bis, date)

    # enregistrer les données enrichies dans un nouveau csv
    data_bis.to_csv(file, index=True)
    #print(data_bis)
    print("Les {} données générées ont bien été ajoutées au fichier {}".format(x*size, file))


def data_generator_x(data, file, x:int=100):
    """ Génère x tirages non-gagnants par tirage gagnant
        les ajoute au dataframe des données
        Puis enregistre le tout dans un fichier (csv)
    Params:
        data (pd.df): dataframe contenant la liste de tirages gagnants
        x (int): nombre de tirages perdants à ajouter à chaque tirage gagnant
        file (str): nom du fichier ou seront enregistrees les donnees
    Returns:
        None
    """
    size = len(data)
    # on ajoute x tirages perdants pour un tirage gagnant
    for k in range(size):
        # récupère la date du tirage gagnant
        date = data["Date"][k]
        for i in range(x) : 
            data = data_generator(data, date)

    # enregistrer les données enrichies dans un nouveau csv
    data.to_csv(file, index=True)
    print("Les {} données générées ont bien été ajoutées au fichier {}".format(x*size, file))
