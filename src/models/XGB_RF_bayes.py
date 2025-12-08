import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBRegressor

from data.final import DefaultScaler, k_, k_fold
from utils.hybrid import get_hyperparams

y_pred = np.array([])
y_test = np.array([])

# Initialize pipeline with weighted hybrid & scaler
pipe = Pipeline(
  [
    ("scaler", DefaultScaler()),
    ("", VotingRegressor([("rf", RandomForestRegressor()), ("xgb", XGBRegressor())])),
  ]
)

hyperparams = get_hyperparams(("xgb", "rf"))

# Cross-validation loop, no scaling because there's a scaler in the Pipeline
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
