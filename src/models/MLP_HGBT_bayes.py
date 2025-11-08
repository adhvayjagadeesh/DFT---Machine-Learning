from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
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
    ("hgbt", HistGradientBoostingRegressor()),
    ("mlp", MLPRegressor())
  ]))
])

hyperparams = {
  "__hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "__hgbt__max_iter": Integer(150, 800),
  "__hgbt__max_leaf_nodes": Integer(20, 50),
  "__hgbt__min_samples_leaf": Integer(10, 40),
  "__hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
  "__hgbt__max_bins": Integer(127, 255),

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