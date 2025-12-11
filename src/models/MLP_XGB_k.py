import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from data.final import k_fold

y_pred = np.array([])
y_test = np.array([])
hybrid = VotingRegressor([("mlp", MLPRegressor()), ("xgb", XGBRegressor())])
for x_train, y_train, x_test_f, y_test_f in k_fold():
  hybrid.fit(x_train, y_train)
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, hybrid.predict(x_test_f)])
