# hybrid_hgbt_svr_bayes.py
# Hybrid ensemble that follows EXACTLY the frameworks & hyperparameter spaces from:
# - HGBT_bayes.py (HistGradientBoostingRegressor)
# - SVR_bayes.py (SVR)
#
# For each fold from data.final.k_fold this runs BayesSearchCV for HGBR and SVR
# using the exact hyperparameter spaces and BayesSearchCV settings from the two files,
# then averages their predictions for that fold and accumulates them into y_pred / y_test.

from sklearn.ensemble import HistGradientBoostingRegressor
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

# HGBT pipeline and hyperparams (exactly as in HGBT_bayes.py)
pipe_hgbt = Pipeline([
    ("scaler", StandardScaler()),
    ("hgbt", HistGradientBoostingRegressor())
])

hyperparams_hgbt = {
    "hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
    "hgbt__max_iter": Integer(150, 800),
    "hgbt__max_leaf_nodes": Integer(20, 50),
    "hgbt__min_samples_leaf": Integer(10, 40),
    "hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
    "hgbt__max_bins": Integer(127, 255),
}

# SVR pipeline and hyperparams (exactly as in SVR_bayes.py)
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

# Iterate over folds (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    # HGBR BayesSearchCV (settings exactly from HGBT_bayes.py; n_jobs=1 as specified)
    bayes_hgbt = BayesSearchCV(
        pipe_hgbt,
        hyperparams_hgbt,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,   # kept exactly as in HGBT_bayes.py
        verbose = 4
    )
    bayes_hgbt.fit(x_train, y_train)

    # SVR BayesSearchCV (settings exactly from SVR_bayes.py; n_jobs=-1 as specified)
    bayes_svr = BayesSearchCV(
        pipe_svr,
        hyperparams_svr,
        cv = k_,
        n_iter = 20,
        n_jobs = -1,
        verbose = 4
    )
    bayes_svr.fit(x_train, y_train)

    # Predict on this fold's test set with both models
    pred_hgbt = bayes_hgbt.predict(x_test_f)
    pred_svr = bayes_svr.predict(x_test_f)

    # Simple hybrid: average the two predictions
    ensemble_pred = (pred_hgbt + pred_svr) / 2.0

    # Combine fold results into global arrays
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, ensemble_pred])

# After the loop, y_test and y_pred contain the concatenated true values and hybrid predictions across folds.
