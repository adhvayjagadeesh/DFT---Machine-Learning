import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data.final import k_fold

y_pred = np.array([])
y_test = np.array([])

rf = RandomForestRegressor()

for x_train, y_train, x_test_f, y_test_f in k_fold():
  rf.fit(x_train, y_train)

  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, rf.predict(x_test_f)])
