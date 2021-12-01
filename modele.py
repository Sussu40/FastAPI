import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data.csv",sep=",", index_col=0)

# Données d'entrainement et de test
X = data[["N1", "N2", "N3", "N4", "N5", "E1","E2"]]
y = data[["Winning Serie"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def entrainement(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, pred)

    # proba = model.predict_proba(X_test)
    # proba[:,1].mean() # 

    return(model)

def metriques(model, X_test, y_test):

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    # ajouter loss etc. 

    proba = model.predict_proba(X_test)
    proba[:,1].mean() # probabilité moyenne de gagner 

    return(accuracy)

def predire(model, x):
    proba = model.predict_proba(x.reshape(1, -1))
    return(proba[0])