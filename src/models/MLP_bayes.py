from skopt import BayesSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from utils.hybrid import make_hyperparam

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("mlp", MLPRegressor())
])

hyperparams = make_hyperparam("mlp")

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_mlp = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with mlp, worsening performance
  )
  bayes_mlp.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_mlp.predict(x_test_f)])