# hybrid_gbt_mlp_bayes.py
# Hybrid ensemble that follows EXACTLY the frameworks & hyperparameter spaces from:
# - GBT_bayes.py (GradientBoostingRegressor)
# - MLP_bayes.py (MLPRegressor)
#
# For each fold from data.final.k_fold this runs BayesSearchCV for GBT and MLP
# using the exact hyperparameter spaces and BayesSearchCV settings from the two files,
# then averages their predictions for that fold and accumulates them into y_pred / y_test.

from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical
from sklearn.neural_network import MLPRegressor

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# GBT pipeline and hyperparams (exactly as in GBT_bayes.py)
pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("", VotingRegressor([
    ("gbt", GradientBoostingRegressor()),
    ("mlp", MLPRegressor())
  ]))
])

hyperparams = {
  "__gbt__n_estimators": Integer(200, 800),
  "__gbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "__gbt__max_depth": Integer(3, 10),
  "__gbt__min_samples_split": Integer(2, 15),
  "__gbt__min_samples_leaf": Integer(1, 10),
  "__gbt__subsample": Real(0.7, 1),
  "__gbt__max_features": Categorical(["sqrt", 0.7, None]),

  "__mlp__hidden_layer_sizes": Integer(100, 500),
  "__mlp__solver": Categorical(["adam", "sgd"]),
  "__mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
  "__mlp__max_iter": Integer(150, 500),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,
  )
  bayes_hybrid.fit(x_train, y_train)

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])