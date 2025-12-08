import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from data.final import k_fold

y_pred = np.array([])
y_test = np.array([])

gbt = GradientBoostingRegressor()

for x_train, y_train, x_test_f, y_test_f in k_fold():
  gbt.fit(x_train, y_train)

  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, gbt.predict(x_test_f)])
