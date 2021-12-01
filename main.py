from fastapi import FastAPI
import csv

app = FastAPI()

#Predict

class Predict():
    metric:int
    name:str


@app.post("/api/predict")
async def predictFromProposal():
    return{"message": "Hello World"}

@app.get("/api/predict")
async def predictListOfWinableNumber():
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

