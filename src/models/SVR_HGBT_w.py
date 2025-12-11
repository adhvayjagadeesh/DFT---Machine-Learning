import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR

from data.final import k_fold, split
from utils.hybrid import VotingRegressor, derive_optimal_weights

y_pred = np.array([])
y_test = np.array([])
hybrid = VotingRegressor(
  [("svr", SVR(max_iter=100000)), ("gbt", HistGradientBoostingRegressor())]
)
for x_train, y_train, x_test_f, y_test_f in k_fold():
  x_train_r, y_train_r, x_train_w, y_train_w = split(x_train, y_train)
  hybrid.fit(x_train_r, y_train_r)
  optimal_weights = derive_optimal_weights(hybrid, x_train_w, y_train_w)
  hybrid.fit(x_train, y_train)
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, hybrid.predict(x_test_f)])
