from fastapi import FastAPI
from app.endpoints import ml_models, utils
from pydantic import BaseModel
import datetime

app = FastAPI()

class Tirage(BaseModel):
    """ Tirage d'EuroMillions : 5 nombres et 2 étoiles, tous entiers """
    N1 : int
    N2 : int
    N3 : int
    N4 : int
    N5 : int
    E1 : int
    E2 : int

@app.get('/')
async def root():
    """ Accueil de l'API, affiche un message de bienvenue """
    return { "Message" : "Bienvenue sur notre Solveur EuroMillions !"}

#Predict
@app.post("/api/predict")
async def predictProbaFromProposal(tirage : Tirage):
    """ Predit la probabilité que le tirage soit gagnant 
    
    Params: 
        tirage (Tirage): le tirage à évaluer (suite de 5 numéros + 2 numéros étoiles)
    
    Returns:
        le dictionnaire contenant la probabilité de victoire du tirage
    """
    # TODO : choix de la date du tirage ? 
    model = ml_models.charger_modele()
    x = [tirage.N1, tirage.N2, tirage.N3, tirage.N4, tirage.N5, tirage.E1, tirage.E2]
    # Prédit la probabilité que la suite x de nombre soit gagnante selon le modèle
    prob = ml_models.predire(model, x)
    return{"Probabilité de gagner": prob}

@app.get("/api/predict/")
async def generateListOfWinableNumbers(objectif : float=0.1):
    """ Génère une suite de numéro qui ont une probabilité importante de gagner 
    
    Params:
        objectif (float): la probabilité minimale visée
    
    Returns:
        le dictionnaire contenant : 
            un message indiquant si un tirage vérifiant la condition a été trouvé
            le tirage correspondant (celui qui vérifie la condition ou le meilleur visité par l'algorithme)
            la probabilité de gagner du tirage
    """
    model = ml_models.charger_modele()
    # Trouve un tirage avec une probabilité de gagner > objectif, ou la plus proche si elle n'a pas été toruvée
    tirage, prob, valide = ml_models.tirer_un_bon(model, objectif)
    message = "Une combinaison probable a été trouvée !" if valide else "Aucune combinaison valable, la meilleure rencontrée est :"
    return{"Resultat": message, "Tirage": str(tirage), "probabilite": prob}

#Model
@app.get("/api/model")
async def GetModelInformations():
    """ Permet d'obtenir les informations techniques du modele 
    
    Returns :
        le dictionnaire contenant l'ensemble des informations importantes
    """
    data_model = utils.lire_data_model()
    return data_model["Valeur"]

@app.put("/api/model")
async def addDataToModel(tirage : Tirage, date : datetime.date, winner : int, gain : int):
    """ Permet d'enrichir le modele d'une donnee supplementaire 
        Vérifie dans un premier lieu que la donnée correspond à un tirage
    
    Params:
        tirage (Tirage): le tirage à enregistrer
        date (datetime.date): la date correspondante
        winner (int): le nombre de gagnants
        gain (int): le gain 
    
    Returns:
        le dictionnaire contenant 
            soit un message d'erreur si le tirage n'est pas valide
            soit un message de succès et l'ensemble des données enregistrées
    """
    # vérification du format des données
    if not utils.check_data(tirage) :
        return {"Erreur": "Impossible d'ajouter une donnée de ce format"}
    # ajout de la donnée au fichier
    utils.add_data(tirage, date, winner, gain)
    return {"Titre": "Nouvelle donnee enregistrée", 
            "tirage": tirage, 
            "date": date,
            "winner": winner,
            "gain": gain,
            }

@app.get("/api/model/train")
async def retrainModel():
    """ Réentraine le modele """
    X_train, X_test, y_train, y_test, infos = ml_models.traitement_donnees("app/data/data.csv")
    ml_models.entrainement(X_train, X_test, y_train, y_test, infos)
    return {"message": "Entrainement effectué"}

