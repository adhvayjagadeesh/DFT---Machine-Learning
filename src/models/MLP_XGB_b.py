import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBRegressor

from data.final import DefaultScaler, k_, k_fold
from utils.hybrid import get_hyperparams

y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline(
  [
    ("scaler", DefaultScaler()),
    ("", VotingRegressor([("xgb", XGBRegressor()), ("mlp", MLPRegressor())])),
  ]
)

hyperparams = get_hyperparams(("xgb", "mlp"))

for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv=k_,
    n_iter=20,
    n_jobs=1,
  )
  bayes_hybrid.fit(x_train, y_train)

  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
