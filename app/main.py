# Data/Path Handling
import logging
import pandas as pd
import pickle
import numpy as np
from pydantic import BaseModel
import os

# Server
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

# Modeling
from xgboost import XGBRegressor

app = FastAPI()

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# Get Pickle path
path = os.path.dirname(os.path.abspath(__file__)).replace('app', 'app\\data')

# Initialize files
model = pickle.load(open(path + '\\model.pickle', 'rb'))
features = pickle.load(open(path + '\\features.pickle', 'rb'))


class Data(BaseModel):
    age: int
    Medu: int
    Fedu: int
    traveltime: int
    studytime: int
    failures: int
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    abscences: int
    G1: int
    G2: int

@app.post("/predict/")
async def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = jsonable_encoder(data)
        for key, value in data_dict.items():
            data_dict[key] = [value]
            to_predict = pd.DataFrame.from_dict(data_dict)
    
    except:
        my_logger.error("Something went wrong!")
        return {"prediction": "error"}
        
    # Create and return prediction
    prediction = model.predict(to_predict)
    return {"prediction": float(prediction[0])}
