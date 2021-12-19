from fastapi import FastAPI
from app.endpoints import ml_models, utils
from pydantic import BaseModel
import datetime

app = FastAPI()

#Predict
class Tirage(BaseModel):
    N1 : int
    N2 : int
    N3 : int
    N4 : int
    N5 : int
    E1 : int
    E2 : int

@app.get('/')
async def root():
    return { "Message" : "Bienvenue sur notre Solveur EuroMillions !"}

#Predit la probabilité que le tirage soit gagnant
@app.post("/api/predict")
async def predictProbaFromProposal(tirage : Tirage):
    # TODO : choix de la date du tirage ? 
    model = ml_models.charger_modele()
    x = [tirage.N1, tirage.N2, tirage.N3, tirage.N4, tirage.N5, tirage.E1, tirage.E2]
    prob = ml_models.predire(model, x)
    return{"Probabilité de gagner": prob}

#Genere une suite de numéro qui ont une probabilité importante de gagner
@app.get("/api/predict/")
async def generateListOfWinableNumbers(objectif : float=0.1):
    model = ml_models.charger_modele()
    tirage, prob, valide = ml_models.tirer_un_bon(model, objectif)
    message = "Une combinaison probable a été trouvée !" if valide else "Aucune combinaison valable, la meilleure rencontrée est :"
    return{"Resultat": message, "Tirage": str(tirage), "probabilite": prob}

#Model
@app.get("/api/model")
async def GetModelInformations():
    """ Permet d'obtenir les informations techniques du modele
    """
    data_model = utils.lire_data_model()
    return data_model["Valeur"]

@app.put("/api/model")
async def addDataToModel(tirage : Tirage, date : datetime.date, winner : int, gain : int):
    """ Permet d'enrichir le modele d'une donnee supplementaire """
    utils.add_data(tirage, date, winner, gain)
    return {"Titre": "Nouvelle donnee enregistrée", 
            "tirage": tirage, 
            "date": date,
            "winner": winner,
            "gain": gain
            }

@app.get("/api/model/train")
async def retrainModel():
    """ Réentraine le modele """
    X_train, X_test, y_train, y_test, infos = ml_models.lecture_donnees("app/data/data.csv")
    ml_models.entrainement(X_train, X_test, y_train, y_test, infos)
    return {"message": "Entrainement effectué"}

