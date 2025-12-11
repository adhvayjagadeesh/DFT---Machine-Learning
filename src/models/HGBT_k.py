import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from data.final import k_fold

y_pred = np.array([])
y_test = np.array([])
hbgt = HistGradientBoostingRegressor()
for x_train, y_train, x_test_f, y_test_f in k_fold():
  hbgt.fit(x_train, y_train)
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, hbgt.predict(x_test_f)])
