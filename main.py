from fastapi import FastAPI
from app.endpoints import ml_models, utils
from pydantic import BaseModel


app = FastAPI()

#Predict
class Tirage(BaseModel):
    #date : date
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
    # TODO : ajouter la date pour la prediction aussi
    model = ml_models.charger_modele()
    x = [tirage.N1, tirage.N2, tirage.N3, tirage.N4, tirage.N5, tirage.E1, tirage.E2]
    prob = ml_models.predire(model, x)
    return{"Probabilité de gagner": prob}

#Genere une suite de numéro qui ont une probabilité importante de gagner
@app.get("/api/predict/{objectif}")
async def generateListOfWinableNumbers(objectif : float):
    model = ml_models.charger_modele()
    tirage, prob = ml_models.tirer_un_bon(model, objectif)
    return{"Tirage": str(tirage), "probabilite": prob}

#Model
@app.get("/api/model")
async def GetModelInformations():
    """ Permet d'obtenir les informations techniques du modele
    """
    data_model = utils.lire_data_model()
    return data_model

@app.put("/api/model")
async def addDataToModel():
    """ Permet d'enrichir le modele d'une donnee supplementaire """

    return{"message": "Hello World"}

@app.get("/api/model/train")
async def retrainModel():
    """ Réentraine le modele """
    X_train, X_test, y_train, y_test = ml_models.lecture_donnees("app/data/data.csv")
    model = ml_models.entrainement(X_train, y_train)
    # enregistrement des infos du modele
    ml_models.save_data(model, X_test, y_test, y_train)
    return{"message": "Entrainement effectué"}

