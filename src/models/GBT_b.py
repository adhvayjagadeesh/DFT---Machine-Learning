import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV

from data.final import DefaultScaler, k_, k_fold
from utils.hybrid import get_hyperparam

y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([("scaler", DefaultScaler()), ("gbt", GradientBoostingRegressor())])

hyperparams = get_hyperparam("gbt")

for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
  bayes_gbt = BayesSearchCV(
    pipe,
    hyperparams,
    cv=k_,
    n_iter=20,
    n_jobs=1,
  )
  bayes_gbt.fit(x_train, y_train)

  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_gbt.predict(x_test_f)])
