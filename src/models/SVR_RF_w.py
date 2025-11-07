from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils.hybrid import WeightedRegressor, derive_optimal_weights
from data.final import k_fold, split
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

hybrid = WeightedRegressor([
  ("svr", SVR()),
  ("rf", RandomForestRegressor())
])

for x_train, y_train, x_test_f, y_test_f in k_fold():
  
  # Resplit the training data for regular training and weighting
  x_train, y_train, x_train_w, y_train_w = split(x_train, y_train)
  hybrid.fit(x_train, y_train)

  optimal_weights = derive_optimal_weights(hybrid, x_train_w, y_train_w)

  # Final prediction and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, hybrid.predict(x_test_f, optimal_weights)])
