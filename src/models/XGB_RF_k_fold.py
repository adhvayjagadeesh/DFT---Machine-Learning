import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

from data.final import k_fold

# Store final predictions
y_pred = np.array([])
y_test = np.array([])

# Initialize weighted hybrid
hybrid = VotingRegressor([("rf", RandomForestRegressor()), ("xgb", XGBRegressor())])

# Cross-validation loop
for x_train, y_train, x_test_f, y_test_f in k_fold():
  hybrid.fit(x_train, y_train)

  # Final prediction
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, hybrid.predict(x_test_f)])
