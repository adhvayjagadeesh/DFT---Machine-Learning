import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBRegressor

from data.final import DefaultScaler, k_, k_fold, split
from utils.hybrid import WeightedRegressor, derive_optimal_weights, get_hyperparams

y_pred = np.array([])
y_test = np.array([])

# Initialize pipeline with adjustable-weight hybrid & scaler
pipe = Pipeline(
  [
    ("scaler", DefaultScaler()),
    ("", WeightedRegressor([("rf", RandomForestRegressor()), ("xgb", XGBRegressor())])),
  ]
)

hyperparams = get_hyperparams(("xgb", "rf"))

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
  optimal_weights = derive_optimal_weights(bayes_hybrid, x_train_w, y_train_w)

  # Final prediction
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
