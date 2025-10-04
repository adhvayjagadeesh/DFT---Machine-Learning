from sklearn.ensemble import RandomForestRegressor
from data.final import k_fold
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# By _f I mean fold
for x_train, y_train, x_test_f, y_test_f in k_fold():
    # Train RF
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)

    # Predict and evaluate
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, rf.predict(x_test_f)])