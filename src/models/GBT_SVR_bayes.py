# hybrid_gbt_svr_bayes.py
# Hybrid ensemble that follows EXACTLY the frameworks & hyperparameter spaces from:
# - GBT_bayes.py
# - SVR_bayes.py
#
# For each fold from data.final.k_fold this runs BayesSearchCV for GradientBoostingRegressor and SVR
# using the exact hyperparameter spaces and BayesSearchCV settings from the two files,
# then averages their predictions for that fold and accumulates them into y_pred / y_test.

from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical
from sklearn.svm import SVR

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# GBT pipeline and hyperparams (exactly as in GBT_bayes.py)
pipe_gbt = Pipeline([
    ("scaler", StandardScaler()),
    ("gbt", GradientBoostingRegressor())
])

hyperparams_gbt = {
    "gbt__n_estimators": Integer(200, 800),
    "gbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
    "gbt__max_depth": Integer(3, 10),
    "gbt__min_samples_split": Integer(2, 15),
    "gbt__min_samples_leaf": Integer(1, 10),
    "gbt__subsample": Real(0.7, 1),
    "gbt__max_features": Categorical(["sqrt", 0.7, None]),
}

# SVR pipeline and hyperparams (exactly as in SVR_bayes.py)
pipe_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(max_iter = 1000000))
])

hyperparams_svr = {
    "svr__C": Real(1e-3, 1e+6, prior="log-uniform"),
    "svr__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
    "svr__degree": Integer(1, 9),
    "svr__epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
    "svr__kernel": Categorical(["linear", "poly", "rbf"]),
}

# Iterate over folds (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    # GBT BayesSearchCV (settings exactly from GBT_bayes.py; n_jobs=1 as specified)
    bayes_gbt = BayesSearchCV(
        pipe_gbt,
        hyperparams_gbt,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,   # kept exactly as in GBT_bayes.py
        verbose = 4
    )
    bayes_gbt.fit(x_train, y_train)

    # SVR BayesSearchCV (settings exactly from SVR_bayes.py; n_jobs=-1 as specified)
    bayes_svr = BayesSearchCV(
        pipe_svr,
        hyperparams_svr,
        cv = k_,
        n_iter = 20,
        n_jobs = -1,  # kept exactly as in SVR_bayes.py
        verbose = 4
    )
    bayes_svr.fit(x_train, y_train)

    # Predict on this fold's test set with both models
    pred_gbt = bayes_gbt.predict(x_test_f)
    pred_svr = bayes_svr.predict(x_test_f)

    # Simple hybrid: average the two predictions
    ensemble_pred = (pred_gbt + pred_svr) / 2.0

    # Combine fold results into global arrays
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, ensemble_pred])

# After the loop, y_test and y_pred contain the concatenated true values and hybrid predictions across folds.