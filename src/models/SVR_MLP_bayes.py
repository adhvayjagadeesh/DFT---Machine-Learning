from sklearn.ensemble import VotingRegressor
from skopt import BayesSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from data.final import k_fold, k_
from skopt.space import Real, Integer, Categorical
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# MLP pipeline and hyperparams (exactly as provided)
pipe_mlp = Pipeline([
  ("scaler", StandardScaler()),
  VotingRegressor([
    ("mlp", MLPRegressor()),
    ("svr", SVR(max_iter = 1000000))
  ])
])

hyperparams_mlp = {
  "__mlp__hidden_layer_sizes": Integer(100, 500),
  "__mlp__solver": Categorical(["adam", "sgd"]),
  "__mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
  "__mlp__max_iter": Integer(150, 500),
  
  "__svr__C": Real(1e-3, 1e+6, prior="log-uniform"),
  "__svr__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
  "__svr__degree": Integer(1, 9),
  "__svr__epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
  "__svr__kernel": Categorical(["linear", "poly", "rbf"]),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe_mlp,
    hyperparams_mlp,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,   # kept exactly as in MLP_bayes.py
    verbose = 4
  )
  bayes_hybrid.fit(x_train, y_train)

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])

# After the loop, y_test and y_pred contain the concatenated true values and hybrid predictions across folds.
