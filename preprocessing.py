import pandas as pd
import numpy as np

data = pd.read_csv("EuroMillions_numbers.csv", sep=";")#, index_col=0)
# Tri par date
data = data.sort_index(axis=0, ascending=True)

# Ajout de la variable cible au dataframe
size = len(data)
y = np.ones(size) # variable cible
data.insert(9, "Winning Serie", y)
# del data["Winner"]
# del data["Gain"]
del data["Date"]

def serie_generator(data):
    # Génération d'une suite de numéros
    boules = list(range(1, 51))
    numbers = np.random.choice(boules, 5, replace=False)
    stars = np.random.choice(boules[:12], 2, replace=False)

    liste = np.zeros(10)
    liste[:5] = numbers #sorted(numbers)
    liste[5:7] = stars #sorted(stars)
    
    # ajout au DF
    df = pd.DataFrame([liste], columns=list(data.columns), index=[str(len(data)+1)])
    data = data.append(df)
    return data

print(data.loc[130])

for i in range(10*len(data)) : 
    data = serie_generator(data)

# créer nouveau csv
data.to_csv("data.csv", index=True)