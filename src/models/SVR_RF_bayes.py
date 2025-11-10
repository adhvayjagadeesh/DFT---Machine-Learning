# hybrid_bayes.py
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from utils.hybrid import make_hyperparams

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# Random Forest pipeline + hyperparams (exactly as in RF_bayes.py)
pipe_rf = Pipeline([
  ("scaler", StandardScaler()),
  ("", VotingRegressor([
    ("svr", SVR(max_iter = 100000)),
    ("rf", RandomForestRegressor())
  ]))
])

hyperparams = make_hyperparams(("svr", "rf"))

# Iterate over the same k_fold generator (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe_rf,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
  )
  bayes_hybrid.fit(x_train, y_train)

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])