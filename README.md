# Prediction des tirage de l'EuroMillion

***AUTEURS : Arthur Molia, Mathieu Surman, Roxane Chatry***

<p>Objectif du projet : exploiter les données des tirages gagnants de l'EuroMillion pour prédire les prochains tirages</p>

<p>Le projet est réalisé en Python, en utilisant FastApi pour l'interface, et scikit-learn pour les modèles prédictifs</p>

<p>Les fonctionnalités suivantes sont implémentées :</p>
<ul>
<li>Prédire la probabilité de gagner d'une proposition de tirage : POST /api/predict</li>
<li>Trouver une proposition de tirage ayant une probabilité élevée de gagner : GET /api/predict</li>
<li>Afficher les informations techniques du modèle : GET /api/model</li>
<li>Ajouter une donnée : PUT /api/model</li>
<li>Réentrainer le modèle : POST /api/model/train</li>
</ul>

## Choix techniques

### Architecture du code
main.py fichier principal

app : dossier racine de l'application, contient tous les packages

<ul>
<li><b>data</b> : fichiers relatifs aux données : <ul>
<li>contient les données sources (EuroMillion_numbers.csv) et les données générées (data.csv)</li>
<li>dossier data_exploration : exploration préalable des données</li>
<li>dossier data_processsing : traitement des données</li>
<li>sorted_100 et not_sorted_200 : anciens fichiers de données générés différemment</li></ul></li>

<li><b>endpoints</b> : modules de fonctions utiles <ul>
<li>ml_models.py : fonctions relatives à la prediction </li>
<li>utils.py : autres fonctions utiles</li></ul></li>

<li><b>static</b> : fichiers générés (sauvegarde du modèle et de ses informations relatives)</li>
</ul>


### Modèle de prédiction

<p><b>Modèle utilisé</b> : RandomForest</p>

<p><b>Données utilisées</b> : le fichier de données ne contenant que des tirages gagnants, on a dû générer des tirages perdant pour compléter les données. Après plusieurs tentatives (avec respectivement 10, 100 et 200 tirages perdants pour un gagnant), on a conservé le fichier contenant 200 tirages perdants pour un tirage gagant, car il représente un bon compromis entre volume de données et qualité des prévisions obtenues. Nous avons tenté de trier les numéros tirés, mais le génération des données était beaucoup plus longue et les changements dans les résultats du modèle n'étaient pas concluants. </p>

<p><b>Variables explicatives</b> : En plus des numéros tirés, on décidé de prendre en compte la date en tant que variable explicative. On la traduit en 4 données : le jour, le mois, l'année, et un delta temporel qualifiant la distance entre les dates. </p>





## Installation de l'application

<p>Telecharger le code</p>
Pour installer les librairies nécessaires: <br/>

>pip install -r requirements.txt

Pour lancer le serveur de FastApi: <br/>

>uvicorn main:app --reload

Tester dans un navigateur sur l'url http://127.0.0.1:8000/docs#/

<br/>
Pour générer un nouveau fichier de données :
<ul>
    <li>nouveau dataset de 100 tirages perdants pour chaque tirage gagnant </li>
    Attention, le fichier de données utilisé précedemment sera écrasé (app/data/data.csv) <br/>
    
>python gen_data.py 
    
<li>nouveau dataset de x tirages enregistré dans le fichier "monfichier.csv" </li>
    
>python gen_data.py x monfichier.csv
    
Attention, la génération de données peut prendre du temps
</ul>
