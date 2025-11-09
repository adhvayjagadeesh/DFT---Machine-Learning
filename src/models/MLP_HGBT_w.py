from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from utils.hybrid import WeightedRegressor, derive_optimal_weights
from data.final import k_fold, split
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

hybrid = WeightedRegressor([
  ("hgbt", HistGradientBoostingRegressor()),
  ("mlp", MLPRegressor())
])

for x_train, y_train, x_test_f, y_test_f in k_fold():
  
  # Resplit the training data for regular training and weighting
  x_train_r, y_train_r, x_train_w, y_train_w = split(x_train, y_train)
  hybrid.fit(x_train_r, y_train_r)
  optimal_weights = derive_optimal_weights(hybrid, x_train_w, y_train_w)

  # Retrain on full training set
  hybrid.fit(x_train)

  # Final prediction and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, hybrid.predict(x_test_f, optimal_weights)])
