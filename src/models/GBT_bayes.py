from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from data.final import k_fold, k_, DefaultScaler
from utils.hybrid import get_hyperparam
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", DefaultScaler()),
  ("gbt", GradientBoostingRegressor())
])

hyperparams = get_hyperparam("gbt")

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_gbt = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
  )
  bayes_gbt.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_gbt.predict(x_test_f)])