import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from skopt import BayesSearchCV

from data.final import DefaultScaler, k_, k_fold, split
from utils.hybrid import VotingRegressor, derive_optimal_weights, get_hyperparams

y_pred = np.array([])
y_test = np.array([])
pipe = Pipeline(
  [
    ("scaler", DefaultScaler()),
    ("", VotingRegressor([("svr", SVR()), ("gbt", GradientBoostingRegressor())])),
  ]
)
hyperparams = get_hyperparams(("svr", "gbt"))
for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
  x_train_t, y_train_t, x_train_w, y_train_w = split(x_train, y_train)
  bayes_hybrid = BayesSearchCV(pipe, hyperparams, cv=k_, n_iter=20, n_jobs=1)
  bayes_hybrid.fit(x_train_t, y_train_t)
  optimal_weights = derive_optimal_weights(
    bayes_hybrid.best_estimator_[1], x_train_w, y_train_w
  )
  pipe.set_params(__weights=optimal_weights, **bayes_hybrid.best_params_)
  pipe.fit(x_train, y_train)
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, pipe.predict(x_test_f)])
