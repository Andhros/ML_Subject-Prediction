import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy
import pickle
import os

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
data = pd.read_csv('C:\\Users\\MASTER\\Documents\\Projects\\Student Dataset\\model\\student-mat.csv', sep=';')
data['target'] = data['G3']
data.drop(columns='G3', inplace=True)
features = data.drop('target', axis=1).select_dtypes([np.number])
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, data['target'], random_state=25)

# está fudido precisa olhar aqui embaixo

# Instantiate model
model = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=RidgeCV())
    ),
    XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=9, n_estimators=1000, nthread=1, objective="reg:squarederror", subsample=0.35000000000000003)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(model.steps, 'random_state', 25)
model.fit(training_features, training_target)
results = model.predict(testing_features)

model.predict(
    data[testing_features.columns]) - data['target']

data['predicted'] = model.predict(data[testing_features.columns])
data['residuals'] = data['target'] - data['predicted']

pickle_names = {'features': features, 'model':model}

for i,j in pickle_names.items():
    pickle_path = os.path.dirname(os.path.abspath(__file__)).replace('model', 'app\\data\\' + i + '.pickle')

    pickle.dump(j, open(pickle_path, 'wb'))
