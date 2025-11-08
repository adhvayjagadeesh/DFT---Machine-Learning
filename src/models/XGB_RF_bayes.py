from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
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
    ("rf", RandomForestRegressor()),
    ("xgb", XGBRegressor())
  ]))
])


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

  "__rf__n_estimators": Integer(100, 1000),
  "__rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
  "__rf__min_samples_split": Integer(2, 20),
  "__rf__min_samples_leaf": Integer(1, 10),
  "__rf__max_features": [1, "sqrt", "log2"],
  "__rf__bootstrap": Categorical([True, False]),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  # Tune XGB (same as your XGB_bayes.py)
  bayes = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
    verbose = 4
  )
  bayes_xgb.fit(x_train, y_train)

  # Tune RF (same as your RF_bayes.py)
  bayes_rf = BayesSearchCV(
    pipe_rf,
    hyperparams_rf,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
    verbose = 4
  )
  bayes_rf.fit(x_train, y_train)

  # Predict and evaluate (simple average blend of the two models)
  y_test = np.concatenate([y_test, y_test_f])
  preds_xgb = bayes_xgb.predict(x_test_f)
  preds_rf = bayes_rf.predict(x_test_f)
  preds_blend = (preds_xgb + preds_rf) / 2.0
  y_pred = np.concatenate([y_pred, preds_blend])