from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# XGB pipeline and hyperparams (exactly as XGB_bayes.py)
pipe = Pipeline([
  ("scaler", StandardScaler()),
  VotingRegressor([
    ("xgb", RandomForestRegressor()),
    ("mlp", MLPRegressor())
  ])
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

  "__mlp__hidden_layer_sizes": Integer(100, 500),
  "__mlp__solver": Categorical(["adam", "sgd"]),
  "__mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
  "__mlp__max_iter": Integer(150, 500),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,
  )
  bayes_hybrid.fit(x_train, y_train)

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
