# Prediction des tirage de l'EuroMillion

## Choix techniques

### Architecture du code


### Modèle de Machine Learning



## Installation de l'application

<p>Telecharger le code</p>
Pour installer les librairies nécessaires: <br/>

>pip install -r requirements.txt

Pour lancer le serveur de FastApi: <br/>

>uvicorn main:app --reload

tester dans un navigateur sur l'url http://127.0.0.1:8000/docs#/

<ul>
Pour générer un nouveau fichier de données :
    <li>nouveau dataset de 100 tirages perdants pour chaque tirage gagnant </li>
    Attention, le fichier de données utilisé précedemment sera écrasé (app/data/data.csv) <br/>
>python gen_data.py 
    <li>nouveau dataset de x tirages enregistré dans le fichier "monfichier.csv" </li>
>python gen_data.py x monfichier.csv
    Attention, la génération de données peut prendre du temps
</ul>
