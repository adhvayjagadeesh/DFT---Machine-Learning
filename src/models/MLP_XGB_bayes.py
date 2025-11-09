from sklearn.ensemble import VotingRegressor
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from utils.hybrid import make_hyperparams

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# XGB pipeline and hyperparams (exactly as XGB_bayes.py)
pipe = Pipeline([
  ("scaler", StandardScaler()),
  VotingRegressor([
    ("xgb", XGBRegressor()),
    ("mlp", MLPRegressor())
  ])
])

hyperparams = make_hyperparams(("xgb", "mlp"))

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
