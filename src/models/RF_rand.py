from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from sklearn.metrics import mean_squared_error

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor())
])

# Educated guesses (from previous runs)
hyperparams = {
    "rf__n_estimators": [100, 600, 700, 1000],
    "rf__max_depth": [None, 25, 50, 75, 100]
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
    rand_rf = RandomizedSearchCV(
        pipe,
        hyperparams,
        cv = k_,
        n_iter = 20,
        n_jobs = -1,
        verbose = 4
    )
    rand_rf.fit(x_train, y_train)

    # Predict and evaluate
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, rand_rf.predict(x_test_f)])