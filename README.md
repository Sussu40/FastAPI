# Prediction des tirage de l'EuroMillion

## Choix techniques

### Architecture du code


### Modèle de Machine Learning



## Installation de l'application

Telecharger le code
Pour installer les librairies nécessaires: 
    pip install -r requirements.txt

Pour lancer le serveur de FastApi:
    uvicorn main:app --reload
tester dans un navigateur sur l'url http://127.0.0.1:8000/docs#/

Pour générer un nouveau fichier de données :
    - nouveau dataset de 100 tirages perdants pour chaque tirage gagnant
    Attention, le fichier de données utilisé précedemment sera écrasé (app/data/data.csv) :
      python gen_data.py 
    - nouveau dataset de x tirages enregistré dans le fichier "monfichier.csv"
      python gen_data.py x monfichier.csv
    Attention, la génération de données peut prendre du temps

