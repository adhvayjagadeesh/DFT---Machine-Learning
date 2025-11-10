from skopt import BayesSearchCV
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from data.final import k_fold, k_, DefaultScaler
from utils.hybrid import get_hyperparams
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", DefaultScaler()),
  ("", VotingRegressor([
    ("rf", RandomForestRegressor()),
    ("hgbt", HistGradientBoostingRegressor())
  ]))
])

hyperparams = get_hyperparams(("rf", "hgbt"))

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
  )
  bayes_hybrid.fit(x_train, y_train)

  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
