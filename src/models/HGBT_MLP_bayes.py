# hybrid_hgbt_mlp_bayes.py
# Hybrid ensemble that follows EXACTLY the frameworks & hyperparameter spaces from:
# - HGBT_bayes.py (HistGradientBoostingRegressor)
# - MLP_bayes.py (MLPRegressor)
#
# For each fold from data.final.k_fold this runs BayesSearchCV for HGBR and MLP
# using the exact hyperparameter spaces and BayesSearchCV settings from the two files,
# then averages their predictions for that fold and accumulates them into y_pred / y_test.

from sklearn.ensemble import HistGradientBoostingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical
from sklearn.neural_network import MLPRegressor

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

# MLP pipeline and hyperparams (exactly as in MLP_bayes.py)
pipe_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor())
])

hyperparams_mlp = {
    "mlp__hidden_layer_sizes": Integer(100, 500),
    "mlp__solver": Categorical(["adam", "sgd"]),
    "mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
    "mlp__max_iter": Integer(150, 500),
}

# Iterate over folds (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    # HGBT BayesSearchCV (settings exactly from HGBT_bayes.py; n_jobs=1 as specified)
    bayes_hgbt = BayesSearchCV(
        pipe_hgbt,
        hyperparams_hgbt,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,   # kept exactly as in HGBT_bayes.py
        verbose = 4
    )
    bayes_hgbt.fit(x_train, y_train)

    # MLP BayesSearchCV (settings exactly from MLP_bayes.py; n_jobs=1 as specified)
    bayes_mlp = BayesSearchCV(
        pipe_mlp,
        hyperparams_mlp,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,   # kept exactly as in MLP_bayes.py
        verbose = 4
    )
    bayes_mlp.fit(x_train, y_train)

    # Predict on this fold's test set with both models
    pred_hgbt = bayes_hgbt.predict(x_test_f)
    pred_mlp = bayes_mlp.predict(x_test_f)

    # Simple hybrid: average the two predictions
    ensemble_pred = (pred_hgbt + pred_mlp) / 2.0

    # Combine fold results into global arrays
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, ensemble_pred])

# After the loop, y_test and y_pred contain the concatenated true values and hybrid predictions across folds.
# (Optional) Evaluate metrics here if desired, e.g.:
# from sklearn.metrics import mean_squared_error, r2_score
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 :", r2_score(y_test, y_pred))
