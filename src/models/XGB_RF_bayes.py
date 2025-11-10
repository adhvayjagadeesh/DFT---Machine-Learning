from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from data.final import k_fold, k_, DefaultScaler
import numpy as np
from utils.hybrid import get_hyperparams

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])


pipe = Pipeline([
  ("scaler", DefaultScaler()),
  ("", VotingRegressor([
    ("rf", RandomForestRegressor()),
    ("xgb", XGBRegressor())
  ]))
])


hyperparams = get_hyperparams(("xgb", "rf"))
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):

  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
  )
  bayes_hybrid.fit(x_train, y_train)

  # Predict and evaluate (simple average blend of the two models)
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])