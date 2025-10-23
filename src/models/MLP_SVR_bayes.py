# hybrid_mlp_svr_bayes.py
# Hybrid ensemble that follows EXACTLY the frameworks & hyperparameter spaces from:
# - MLP_bayes.py
# - SVR_bayes.py
#
# For each fold from data.final.k_fold this runs BayesSearchCV for MLPRegressor and SVR
# using the exact hyperparameter spaces and BayesSearchCV settings from the two files,
# then averages their predictions for that fold and accumulates into y_pred / y_test.

from skopt import BayesSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from data.final import k_fold, k_
from skopt.space import Real, Integer, Categorical
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# MLP pipeline and hyperparams (exactly as provided)
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

# SVR pipeline and hyperparams (exactly as provided)
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
    # MLP BayesSearchCV (exact settings from MLP_bayes.py)
    bayes_mlp = BayesSearchCV(
        pipe_mlp,
        hyperparams_mlp,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,   # kept exactly as in MLP_bayes.py
        verbose = 4
    )
    bayes_mlp.fit(x_train, y_train)

    # SVR BayesSearchCV (exact settings from SVR_bayes.py)
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
    pred_mlp = bayes_mlp.predict(x_test_f)
    pred_svr = bayes_svr.predict(x_test_f)

    # Simple hybrid: average the two predictions
    ensemble_pred = (pred_mlp + pred_svr) / 2.0

    # Combine fold results into global arrays
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, ensemble_pred])

# After the loop, y_test and y_pred contain the concatenated true values and hybrid predictions across folds.
