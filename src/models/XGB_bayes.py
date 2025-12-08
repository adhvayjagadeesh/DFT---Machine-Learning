import numpy as np
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBRegressor

from data.final import DefaultScaler, k_, k_fold
from utils.hybrid import get_hyperparam

y_pred = np.array([])
y_test = np.array([])

# Initialize pipeline with scaler and XGB
pipe = Pipeline([("scaler", DefaultScaler()), ("xgb", XGBRegressor())])

hyperparams = get_hyperparam("xgb")

# K-Fold loop, no scaling because there's a scaler in the pipeline
for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
  # Bayesian optimization
  bayes_xgb = BayesSearchCV(
    pipe,
    hyperparams,
    cv=k_,
    n_iter=20,
    n_jobs=1,
  )
  bayes_xgb.fit(x_train, y_train)

  # Final prediction
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_xgb.predict(x_test_f)])
