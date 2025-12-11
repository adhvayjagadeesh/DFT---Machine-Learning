import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from skopt import BayesSearchCV
from xgboost import XGBRegressor

from data.final import DefaultScaler, k_, k_fold, split
from utils.hybrid import VotingRegressor, derive_optimal_weights, get_hyperparams

y_pred = np.array([])
y_test = np.array([])

# Initialize pipeline with hybrid & scaler
pipe = Pipeline(
  [
    ("scaler", DefaultScaler()),
    ("", VotingRegressor([("svr", SVR()), ("xgb", XGBRegressor())])),
  ]
)

hyperparams = get_hyperparams(("svr", "xgb"))

# K-Fold loop, no scaling because there's a scaler in the pipeline
for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
  # Resplit training data for tuning and weighting
  x_train_t, y_train_t, x_train_w, y_train_w = split(x_train, y_train)

  # Bayesian optimization with 1st training split
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv=k_,
    n_iter=20,
    n_jobs=1,
  )
  bayes_hybrid.fit(x_train_t, y_train_t)

  # Derive optimal weight with 2nd training split
  optimal_weights = derive_optimal_weights(
    bayes_hybrid.best_estimator_[1], x_train_w, y_train_w
  )

  # Update weights + hyperparams to optimal
  pipe.set_params(__weights=optimal_weights, **bayes_hybrid.best_params_)

  # Refit on all training data
  pipe.fit(x_train, y_train)

  # Final prediction
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, pipe.predict(x_test_f)])
