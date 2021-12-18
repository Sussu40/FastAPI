# from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import os
import csv
# import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import datetime

def creationData(nbLoser):
    """
    Function creationFichiers - Crée le fichier data.csv qui
    contient nbLoser tirages perdants et les tirages gagnants
    
    ---------------------------------------------------------
    
    Author     : Arthur Molia - <moliaarthu@eisti.eu>
    Parameters : nbLoser : nombre de perdants par date
    ---------------------------------------------------------
    """
    em = pd.read_csv('EuroMillions_numbers.csv',sep=";", index_col=0)

    # Creation des fichiers loser et data
    os.system("touch data.csv")

    with open('data.csv', 'w', newline='') as data:
        writerData  = csv.writer(data)

        # Ecrit la ligne de labels
        writerData.writerow(["i;Date;N1;N2;N3;N4;N5;E1;E2;Winner;Gain"])

        cptTot = 0

        # Génération d'une suite de numéros
        boules = list(range(1, 51))
        for row in em.iterrows():
            # Affiche l'avancée de la fonction
            print(row[0])

            writerData.writerow(
                [str(cptTot) + ";" + str(row[0]) + ";" + str(row[1]["N1"]) + ";" + str(row[1]["N2"]) + ";" + str(row[1]["N3"]) + ";" + str(row[1]["N4"]) + ";" + str(row[1]["N5"]) + ";" + str(row[1]["E1"]) + ";" + str(row[1]["E2"]) + ";" + str(row[1]["Winner"]) + ";" + str(row[1]["Gain"])]
            )
            cptTot += 1

            cpt = 0

            # Réalise nbLoser tirages
            while cpt < nbLoser:
                #Génération aléatoire du tirage
                numbers = sorted(np.random.choice(boules, 5, replace=False))
                stars = sorted(np.random.choice(boules[:12], 2, replace=False))

                # Vérifie que le tirage ne soit pas gagnant
                if not(numbers == [row[1]["N1"], row[1]["N2"], row[1]["N3"], row[1]["N4"], row[1]["N5"]] and stars == [row[1]["E1"], row[1]["E2"]]):
                    # Ajoute la ligne correspondant au tirage dans le fichier loser.csv
                    writerData.writerow(
                        [str(cptTot) + ";" + str(row[0]) + ";" + str(numbers[0]) + ";" + str(numbers[1]) + ";" + str(numbers[2]) + ";" + str(numbers[3]) + ";" + str(numbers[4]) + ";" + str(stars[0]) + ";" + str(stars[1]) + ";0;0"]
                    )
                    
                    cpt += 1
                    cptTot += 1

    data.close()

def preprocessing():
    """
    Function preprocessing - prépare les données pour pouvoir
    les utiliser plus tard dans le modèle
    
    ---------------------------------------------------------
    
    Author     : Arthur Molia - <moliaarthu@eisti.eu>
    Parameters : 
    ---------------------------------------------------------
    """
    data = pd.read_csv("data.csv",sep=";", index_col=0)

    # Ajout de la variable cible aux dataframes
    size = len(data)
    y = np.empty(size) # variable cible
    data.insert(9, "Winning Serie", y)

    for i in range(size):
        if data.at[i,"Gain"] == 0:
            # print(row[1]["Gain"])
            data.at[i,"Winning Serie"] = 0.0
            # print(row[1]["Winning Serie"])
        else:
            data.at[i,"Winning Serie"] = 1.0

    # transformation de la date en variable quantitative
    dates = [data["Date"][i].split("-") for i in range(len(data))]
    dates = np.array(dates).T
    # ajout au df
    data.insert(10, "Annee", dates[0])
    data.insert(11, "Mois", dates[1])
    data.insert(12, "Jour", dates[2])

    # print(data.shape)
    # print(data.head())
    # print(data["Winning Serie"].value_counts())


    # liste des variables explicatives / variable cible
    X = data[["N1", "N2", "N3", "N4", "N5", "E1", "E2", "Annee", "Mois", "Jour"]]
    y = data[["Winning Serie"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def entrainement(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, np.ravel(y_train))

    # enregistrer le modèle
    file = "app/static/ml_model.joblib.pkl"
    joblib.dump(model, file, compress=9)

    return(model)

def predire(model, x):
    """ Fonction qui prédit la probabilité qu'une suite x de nombre soit gagnante, 
        selon le modèle model

    Args:
        model (type??): le modèle de prédiction
        x (list of int): la série d'entiers correspondant a une grille jouée à l'euromilion

    Returns:
        float: la probabilité que la suite jouée x soit gagnante

    """
    # ajout de la date au tirage
    today = datetime.date.today()
    annee = today.strftime("%Y")
    mois = today.strftime("%m")
    jour = today.strftime("%d")
    x = np.append(x, [annee, mois, jour])
        
    proba = model.predict_proba(x.reshape(1, -1))
    return(proba[0][1])

def serie_generator():
    """ Génère un tirage aléatoire """
    # Génération d'une suite de numéros
    boules = list(range(1, 51))
    numbers = np.random.choice(boules, 5, replace=False)
    stars = np.random.choice(boules[:12], 2, replace=False)

    liste = np.zeros(7)
    liste[:5] = sorted(numbers)
    liste[5:7] = sorted(stars)
    # liste[:5] = numbers
    # liste[5:7] = stars

    return(liste)

def tirer_un_bon(model, prob_min):
    """ Fonction qui trouve un gagnant probable 

    Args : 
        model (obj): le modèle de prédiction
        prob_min (float): le minimum de probabilité de gagner souhaitée
    
    Returns:
        list of int: le tirage de probabilité supérieur à prob_min 
            s'il a été trouvé
            sinon le tirage le plus probable visité
    """

    Kmax = 10**5 # Nombre d'irérations max
    k = 0 # compteur d'itérations
    tirage_best = [] # enregistrement du meilleur tirage visité
    prob_best = 0 # et de sa probabilité de gagner

    # arret au bout de k itérations si impossible de trouver une proba > prob_min
    # retenir le meilleur tirage calculé dans ce cas avec sa proba
    while k < Kmax:
        tirage = serie_generator()
        prob = predire(model=model, x=tirage)
        if prob >= prob_min:
            return tirage, prob
        if prob > prob_best:
            tirage_best = tirage
            prob_best = prob
        k += 1
    
    return tirage_best, prob_best

def charger_modele():
    model = joblib.load("app/static/ml_model.joblib.pkl")
    return model

#
#
#
#
# Temporary main
#
#
#
#
# creationData(100)
X_train, X_test, y_train, y_test = preprocessing()
model = entrainement(X_train, y_train)
# model = charger_modele()

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:")
print(sklearn.metrics.confusion_matrix(y_test, y_pred))

tirage, prob = tirer_un_bon(model, 0.05)

print("tirage : ", tirage)
print("prob : ", prob)
