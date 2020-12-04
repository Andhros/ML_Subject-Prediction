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


# está fudido precisa olhar aqui embaixo
# @app.post("/predict/")
# async def predict(data: Data):
#     # return data
#     try:
#         # Extract data in correct order
#         to_predict = [data[feature] for feature in features]

#         # Create and return prediction
#         prediction = model.predict(np.array(to_predict).reshape(1, -1))
#         return {"prediction": int(prediction[0])}

#     except:
#         my_logger.error("Something went wrong!")
#         return {"prediction": "error"}

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

@app.post("/predict/")
def predict(data: Data):
    # Extract data in correct order
    data_dict = jsonable_encoder(data)
    for key, value in data_dict.items():
        data_dict[key] = [value]
        to_predict = pd.DataFrame.from_dict(data_dict)

    # Create and return prediction
    prediction = model.predict(to_predict)
    return prediction[0]
