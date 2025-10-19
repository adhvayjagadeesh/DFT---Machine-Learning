from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor())
])

pipe_gbt = Pipeline([
    ("scaler", StandardScaler()),
    ("gbt", GradientBoostingRegressor())
])

# XGBoost hyperparameters (unchanged)
hyperparams_xgb = {
    "xgb__max_depth": Integer(3, 10),
    "xgb__min_child_weight": Integer(1, 8),
    "xgb__learning_rate": Real(1e-2, 0.2, prior="log-uniform"),
    "xgb__n_estimators": Integer(100, 800),
    "xgb__subsample": Real(0.7, 1.0),
    "xgb__colsample_bytree": Real(0.6, 1.0),
    "xgb__reg_alpha": Real(1e-5, 0.5, prior="log-uniform"),
    "xgb__reg_lambda": Real(0.5, 2.0, prior="log-uniform"),
    "xgb__gamma": Real(0.0, 2.0),
    "xgb__tree_method": Categorical(["hist", "approx"]),
}

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

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    # Tune XGB (same as your XGB_bayes.py)
    bayes_xgb = BayesSearchCV(
        pipe_xgb,
        hyperparams_xgb,
        cv = k_,
        n_iter = 20,
        n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
        verbose = 4
    )
    bayes_xgb.fit(x_train, y_train)

    # Tune GBT (same style as your GBT_bayes.py)
    bayes_gbt = BayesSearchCV(
        pipe_gbt,
        hyperparams_gbt,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,
        verbose = 4
    )
    bayes_gbt.fit(x_train, y_train)

    # Predict and evaluate (simple average blend of the two models)
    y_test = np.concatenate([y_test, y_test_f])
    preds_xgb = bayes_xgb.predict(x_test_f)
    preds_gbt = bayes_gbt.predict(x_test_f)
    preds_blend = (preds_xgb + preds_gbt) / 2.0
    y_pred = np.concatenate([y_pred, preds_blend])
