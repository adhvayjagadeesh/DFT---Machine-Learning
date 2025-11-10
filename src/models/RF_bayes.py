from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from data.final import k_fold, k_, DefaultScaler
import numpy as np
from utils.hybrid import get_hyperparam

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", DefaultScaler()),
  ("rf", RandomForestRegressor())
])

hyperparams = get_hyperparam("rf")

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_rf = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
  )
  bayes_rf.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_rf.predict(x_test_f)])