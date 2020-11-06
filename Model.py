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
import plotly.express as px
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.target import FeatureCorrelation
from yellowbrick.regressor import PredictionError

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:\\Users\\MASTER\\Documents\\Student Dataset\\student-mat.csv', sep=';')
tpot_data['target'] = tpot_data['G3']
tpot_data.drop(columns='G3', inplace=True)
features = tpot_data.drop('target', axis=1).select_dtypes([np.number])
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=25)

# Average CV score on the training set was: 0.8713511039501057
model = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=RidgeCV())
    ),
    XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=9, n_estimators=1000, nthread=1, objective="reg:squarederror", subsample=0.35000000000000003)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(model.steps, 'random_state', 25)
YB = ResidualsPlot(model, hist=True, qqplot=False)

# YB = PredictionError(model)
YB.fit(training_features, training_target)
results = YB.predict(testing_features)

YB.predict(
    tpot_data[testing_features.columns]) - tpot_data['target']

tpot_data['predicted'] = YB.predict(tpot_data[testing_features.columns])
tpot_data['residuals'] = tpot_data['target'] - tpot_data['predicted']


YB.show()

linplot = px.line(
    x=tpot_data.index.values, y=[
        tpot_data['target'], tpot_data['predicted']]
)
linplot.show()

linplot2 = px.line(
    x=tpot_data.index.values, y=[
        tpot_data['residuals'], tpot_data['residuals'].rolling(5).mean()]
)
linplot2.show()
