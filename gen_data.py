from app.data.data_processing.preprocessing import data_generator_x
import pandas as pd
import sys

file = "app/data/data.csv"
# premier argument : nombre de données générées
if len(sys.argv) >= 2:
    x = int(sys.argv[1])
    # deuxième argument : fichier cible
    if len(sys.argv) >= 3:
        file = sys.argv[2]        
    data_generator_x(data=pd.read_csv("app/data/EuroMillions_numbers.csv", sep=";"), file=file, x=x)
else : 
    data_generator_x(data=pd.read_csv("app/data/EuroMillions_numbers.csv", sep=";"), file=file)