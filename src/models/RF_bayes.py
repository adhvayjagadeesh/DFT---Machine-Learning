from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor())
])

hyperparams = {
    "rf__n_estimators": Integer(100, 1000),
    "rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
    "rf__min_samples_split": Integer(2, 20),
    "rf__min_samples_leaf": Integer(1, 10),
    "rf__max_features": [1, "sqrt", "log2"],
    "rf__bootstrap": Categorical([True, False]),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    rand_rf = BayesSearchCV(
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