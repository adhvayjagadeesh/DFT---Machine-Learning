from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from data.final import k_fold, k_, DefaultScaler
import numpy as np
from sklearn.svm import SVR
from utils.hybrid import get_hyperparams

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# GBT pipeline and hyperparams (exactly as in GBT_bayes.py)
pipe = Pipeline([
  ("scaler", DefaultScaler()),
  ("", VotingRegressor([
    ("hgbt", HistGradientBoostingRegressor()),
    ("svr", SVR(max_iter = 100000))
  ]))
])

hyperparams = get_hyperparams(("svr", "hgbt"))

pipe = Pipeline([
  ("scaler", DefaultScaler()),
  VotingRegressor([
    ("svr", SVR(max_iter = 100000)),
    ("hgbt", HistGradientBoostingRegressor())
  ])
])

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