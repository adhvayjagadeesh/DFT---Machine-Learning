from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from skopt import BayesSearchCV
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
  ("", VotingRegressor([
    ("xgb", RandomForestRegressor()),
    ("svr", SVR(max_iter = 1000000))
  ]))
])

hyperparams = {
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

  "svr__C": Real(1e-3, 1e+6, prior="log-uniform"),
  "svr__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
  "svr__degree": Integer(1, 9),
  "svr__epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
  "svr__kernel": Categorical(["linear", "poly", "rbf"]),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,
    verbose = 4
  )
  bayes_hybrid.fit(x_train, y_train)

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
