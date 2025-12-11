import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data.final import k_fold, split
from utils.hybrid import VotingRegressor, derive_optimal_weights

# Store final predictions
y_pred = np.array([])
y_test = np.array([])

# Initialize weighted hybrid
hybrid = VotingRegressor([("rf", RandomForestRegressor()), ("xgb", XGBRegressor())])

# K-Fold loop
for x_train, y_train, x_test_f, y_test_f in k_fold():
  # Resplit training data for regular training and weighting
  x_train_r, y_train_r, x_train_w, y_train_w = split(x_train, y_train)

  # Regular fit on 1st training split
  hybrid.fit(x_train_r, y_train_r)

  # Derive optimal weight with 2nd training split
  optimal_weights = derive_optimal_weights(hybrid, x_train_w, y_train_w)

  # Update weights to optimal
  hybrid.set_params(weights=optimal_weights)

  # Refit on all training data
  hybrid.fit(x_train, y_train)

  # Final prediction
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, hybrid.predict(x_test_f)])
