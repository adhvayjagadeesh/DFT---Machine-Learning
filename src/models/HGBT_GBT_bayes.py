from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
from utils.hybrid import make_hyperparams
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("", VotingRegressor([
    ("gbt", GradientBoostingRegressor()),
    ("hgbt", HistGradientBoostingRegressor())
  ]))
])

hyperparams = make_hyperparams(("gbt", "hgbt"))

for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv=k_,
    n_iter=20,
    n_jobs=1,
    verbose=4
  )
  bayes_hybrid.fit(x_train, y_train)

  # Store predictions and ground truth
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
