# hybrid_xgb_svr_bayes.py
# Hybrid ensemble that follows EXACTLY the frameworks & hyperparameter spaces from:
# - XGB_bayes.py
# - SVR_bayes.py
#
# It runs BayesSearchCV for XGBRegressor and SVR on each fold from data.final.k_fold,
# then combines predictions for each fold by simple averaging (XGB + SVR) and accumulates them into y_pred / y_test.

from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# XGB pipeline and hyperparams (exactly as in XGB_bayes.py)
pipe_xgb = Pipeline([
  ("scaler", StandardScaler()),
  ("xgb", XGBRegressor())
])

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
  # XGB BayesSearchCV (settings exactly from XGB_bayes.py; n_jobs=1 as specified)
  bayes_xgb = BayesSearchCV(
    pipe_xgb,
    hyperparams_xgb,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,   # kept exactly as in XGB_bayes.py to avoid XGB thread conflicts
    verbose = 4
  )
  bayes_xgb.fit(x_train, y_train)

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
  pred_xgb = bayes_xgb.predict(x_test_f)
  pred_svr = bayes_svr.predict(x_test_f)

  # Simple hybrid: average the two predictions
  ensemble_pred = (pred_xgb + pred_svr) / 2.0

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, ensemble_pred])

# At this point y_test and y_pred contain the concatenated true and ensemble predictions across folds.
# (Optional) Evaluate metrics here if desired, e.g.:
# from sklearn.metrics import mean_squared_error, r2_score
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 :", r2_score(y_test, y_pred))
