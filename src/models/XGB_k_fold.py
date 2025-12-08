import numpy as np
from xgboost import XGBRegressor

from data.final import k_fold

y_pred = np.array([])
y_test = np.array([])

# Initialize XGB
xgb = XGBRegressor()

# K-Fold loop
for x_train, y_train, x_test_f, y_test_f in k_fold():
  xgb.fit(x_train, y_train)

  # Final prediction
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, xgb.predict(x_test_f)])
