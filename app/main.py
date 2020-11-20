# Data/Path Handling
import logging
import pickle
import numpy as np
from pydantic import BaseModel
import os

# Server
import uvicorn
from fastapi import FastAPI

# Modeling
from xgboost import XGBRegressor

app = FastAPI()

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# Get Pickle path
path = os.path.dirname(os.path.abspath(__file__)).replace('model', 'app\\data')

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


@app.post("/predict")
def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = [data_dict[feature] for feature in features]

        # Apply one-hot encoding
        encoded_features = list(enc.transform(
            np.array(to_predict[-2:]).reshape(1, -1))[0])
        to_predict = np.array(to_predict[:-2] + encoded_features)

        # Create and return prediction
        prediction = model.predict(to_predict.reshape(1, -1))
        return {"prediction": int(prediction[0])}

    except:
        my_logger.error("Something went wrong!")
        return {"prediction": "error"}
