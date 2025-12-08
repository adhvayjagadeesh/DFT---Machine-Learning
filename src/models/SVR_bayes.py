import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from skopt import BayesSearchCV

from data.final import DefaultScaler, k_, k_fold
from utils.hybrid import get_hyperparam

y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline(
  [
    ("scaler", DefaultScaler()),
    ("svr", SVR(max_iter=100000)),
  ]
)

hyperparams = get_hyperparam("svr")

for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
  bayes_svr = BayesSearchCV(
    pipe,
    hyperparams,
    cv=k_,
    n_jobs=1,
    n_iter=20,
  )
  bayes_svr.fit(x_train, y_train)

  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_svr.predict(x_test_f)])
