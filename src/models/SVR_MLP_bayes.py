from sklearn.ensemble import VotingRegressor
from skopt import BayesSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from data.final import k_fold, k_, DefaultScaler
import numpy as np
from utils.hybrid import get_hyperparams

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", DefaultScaler()),
  ("", VotingRegressor([
    ("mlp", MLPRegressor()),
    ("svr", SVR(max_iter = 100000))
  ]))
])

hyperparams = get_hyperparams(("svr", "mlp"))

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

# After the loop, y_test and y_pred contain the concatenated true values and hybrid predictions across folds.
