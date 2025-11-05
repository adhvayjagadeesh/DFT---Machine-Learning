from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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

pipe_hgbt = Pipeline([
  ("scaler", StandardScaler()),
  ("hgbt", HistGradientBoostingRegressor())
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

  # Tune HGBT (same style as your HGBT_bayes.py)
  bayes_hgbt = BayesSearchCV(
    pipe_hgbt,
    hyperparams_hgbt,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,
    verbose = 4
  )
  bayes_hgbt.fit(x_train, y_train)

  # Predict and evaluate (simple average blend of the two models)
  y_test = np.concatenate([y_test, y_test_f])
  preds_xgb = bayes_xgb.predict(x_test_f)
  preds_hgbt = bayes_hgbt.predict(x_test_f)
  preds_blend = (preds_xgb + preds_hgbt) / 2.0
  y_pred = np.concatenate([y_pred, preds_blend])
