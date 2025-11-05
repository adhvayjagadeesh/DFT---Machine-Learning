from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("xgb", XGBRegressor())
])

# I tried the "dart" booster but it takes years...
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
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_xgb = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
    verbose = 4
  )
  bayes_xgb.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_xgb.predict(x_test_f)])