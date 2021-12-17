import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def addDataToModel(date: str,n1: int,n2: int,n3: int,n4: int,n5: int,e1: int,e2: int,winner: int,gain: int, df):
    df2 = pd.DataFrame([[n1,n2,n3,n4,n5,e1,e2,winner,gain]],columns=list(df.columns), index=[date])
    df = df.append(df2)
    return df

#lecture du csv
df = pd.read_csv('EuroMillions_numbers.csv', sep = ';', index_col=0)
print(df)

#ajout d'une donnée
df = addDataToModel("2022-01-01",0,0,0,0,0,0,0,0,00000000,df)

#trie du csv
df = df.sort_index(axis=0,ascending=True)
print(df)

#génération de random

balls = list(range(1,51))
numbers = np.random.choice(balls,5,replace= False) 

star = list(range(1,13))
stars = np.random.choice(star,2,replace= False)

#Compte le nombre d'occurrence
# N1 = df["N1"].value_counts()
# N2 = df["N2"].value_counts()
# N3 = df["N3"].value_counts()
# N4 = df["N4"].value_counts()
# N5 = df["N5"].value_counts()
# E1 = df["E1"].value_counts()
# E2 = df["E2"].value_counts()

# N = N1 + N2 + N3 + N4 + N5
# E = E1 + E2

# plt.bar(N.index,N.values)
# plt.show()

# plt.bar(E.index,E.values)
# plt.show()





