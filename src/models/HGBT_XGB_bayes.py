from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
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
  ("", VotingRegressor([
    ("xgb", XGBRegressor()),
    ("hgbt", HistGradientBoostingRegressor())
  ]))
])


# XGBoost hyperparameters (unchanged)
hyperparams = {
  "__xgb__max_depth": Integer(3, 10),
  "__xgb__min_child_weight": Integer(1, 8),
  "__xgb__learning_rate": Real(1e-2, 0.2, prior="log-uniform"),
  "__xgb__n_estimators": Integer(100, 800),
  "__xgb__subsample": Real(0.7, 1.0),
  "__xgb__colsample_bytree": Real(0.6, 1.0),
  "__xgb__reg_alpha": Real(1e-5, 0.5, prior="log-uniform"),
  "__xgb__reg_lambda": Real(0.5, 2.0, prior="log-uniform"),
  "__xgb__gamma": Real(0.0, 2.0),
  "__xgb__tree_method": Categorical(["hist", "approx"]),

  "__hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "__hgbt__max_iter": Integer(150, 800),
  "__hgbt__max_leaf_nodes": Integer(20, 50),
  "__hgbt__min_samples_leaf": Integer(10, 40),
  "__hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
  "__hgbt__max_bins": Integer(127, 255),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe_xgb,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
  )
  bayes_hybrid.fit(x_train, y_train)

  # Predict and evaluate (simple average blend of the two models)
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
