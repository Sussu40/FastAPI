from fastapi import FastAPI

app = FastAPI()

#Predict

class Predict():
    metric:int
    name:str

#Predit la probabilité que le tirage soit gagnant
@app.post("/api/predict")
async def predictProbaFromProposal():
    return{"message": "Hello World"}

#Genere une suite de numéro qui ont une probabilité importante de gagner
@app.get("/api/predict")
async def generateListOfWinableNumbers():
    return{"message": "Hello World"}

#Model

@app.get("/api/model")
async def GetModelInformations():
    return{"message": "Hello World"}

@app.put("/api/model")
async def addDataToModel():
    return{"message": "Hello World"}

@app.get("/api/model/train")
async def retrainModel():
    return{"message": "Hello World"}

