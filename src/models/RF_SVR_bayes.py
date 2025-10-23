# hybrid_bayes.py
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Integer, Real, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# Random Forest pipeline + hyperparams (exactly as in RF_bayes.py)
pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor())
])

hyperparams_rf = {
    "rf__n_estimators": Integer(100, 1000),
    "rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
    "rf__min_samples_split": Integer(2, 20),
    "rf__min_samples_leaf": Integer(1, 10),
    "rf__max_features": [1, "sqrt", "log2"],
    "rf__bootstrap": Categorical([True, False]),
}

# SVR pipeline + hyperparams (exactly as in SVR_bayes.py)
pipe_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(max_iter = 1000000))  # keep the large max_iter as provided
])

hyperparams_svr = {
    "svr__C": Real(1e-3, 1e+6, prior="log-uniform"),
    "svr__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
    "svr__degree": Integer(1, 9),
    "svr__epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
    "svr__kernel": Categorical(["linear", "poly", "rbf"]),
}

# Iterate over the same k_fold generator (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    # Bayesian search for Random Forest
    bayes_rf = BayesSearchCV(
        pipe_rf,
        hyperparams_rf,
        cv = k_,
        n_iter = 20,
        n_jobs = -1,
        verbose = 4
    )
    bayes_rf.fit(x_train, y_train)

    # Bayesian search for SVR
    bayes_svr = BayesSearchCV(
        pipe_svr,
        hyperparams_svr,
        cv = k_,
        n_iter = 20,
        n_jobs = -1,
        verbose = 4
    )
    bayes_svr.fit(x_train, y_train)

    # Predict on the fold's test set with both models
    pred_rf = bayes_rf.predict(x_test_f)
    pred_svr = bayes_svr.predict(x_test_f)

    # Simple hybrid: average the two model predictions (changeable)
    ensemble_pred = (pred_rf + pred_svr) / 2.0

    # Combine fold results into global arrays
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, ensemble_pred])

# Optionally: at this point you can evaluate y_test vs y_pred (e.g., compute MSE, R2, etc.)
# But per framework above, the file stops after aggregating predictions from all folds.
