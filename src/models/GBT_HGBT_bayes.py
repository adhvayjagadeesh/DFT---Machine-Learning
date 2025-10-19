from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe_gbt = Pipeline([
    ("scaler", StandardScaler()),
    ("gbt", GradientBoostingRegressor())
])

pipe_hgbt = Pipeline([
    ("scaler", StandardScaler()),
    ("hgbt", HistGradientBoostingRegressor())
])

# GradientBoosting hyperparameters (unchanged)
hyperparams_gbt = {
    "gbt__n_estimators": Integer(200, 800),
    "gbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
    "gbt__max_depth": Integer(3, 10),
    "gbt__min_samples_split": Integer(2, 15),
    "gbt__min_samples_leaf": Integer(1, 10),
    "gbt__subsample": Real(0.7, 1),
    "gbt__max_features": Categorical(["sqrt", 0.7, None]),
}

# HistGradientBoosting hyperparameters (unchanged)
hyperparams_hgbt = {
    "hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
    "hgbt__max_iter": Integer(150, 800),
    "hgbt__max_leaf_nodes": Integer(20, 50),
    "hgbt__min_samples_leaf": Integer(10, 40),
    "hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
    "hgbt__max_bins": Integer(127, 255),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    # Tune GBT (same as your GBT_bayes.py)
    bayes_gbt = BayesSearchCV(
        pipe_gbt,
        hyperparams_gbt,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,  # same as your GBT_bayes.py
        verbose = 4
    )
    bayes_gbt.fit(x_train, y_train)

    # Tune HGBT (same as your HGBT_bayes.py)
    bayes_hgbt = BayesSearchCV(
        pipe_hgbt,
        hyperparams_hgbt,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,  # same as your HGBT_bayes.py
        verbose = 4
    )
    bayes_hgbt.fit(x_train, y_train)

    # Predict and evaluate (simple average blend of the two models)
    y_test = np.concatenate
