import numpy as np
import pandas as pd
from tpot import TPOTRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn.metrics
import timeit

df_mat = pd.read_csv('student-mat.csv', sep=';')

X = df_mat.drop(columns='G3').select_dtypes([np.number])

y = df_mat['G3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 666)
# instantiate TPOT
tpot = TPOTRegressor(verbosity=3,  
                      random_state=25, 
                      n_jobs=-1,
                      scoring='r2', 
                      generations=100, 
                      population_size=30,
                      early_stop = 3,
                      memory = 'auto')
times = []
scores = []
winning_pipes = []

# run 3 iterations
for x in range(3):
    start_time = timeit.default_timer()
    tpot.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    times.append(elapsed)
    winning_pipes.append(tpot.fitted_pipeline_)
    scores.append(tpot.score(X_test, y_test))
    tpot.export('tpot_stu_mat.py')
# output results
times = [time/60 for time in times]
print('Times:', times)
print('Scores:', scores)
print('Winning pipelines:', winning_pipes)