from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
from utils.hybrid import make_hyperparams
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# For recording the derived weights
weights = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("", VotingRegressor([
    ("rf", RandomForestRegressor()),
    ("gbt", GradientBoostingRegressor())
  ]))
])

hyperparams = make_hyperparams(("rf", "gbt"))

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
  )
  bayes_hybrid.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
