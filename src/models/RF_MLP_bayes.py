# hybrid_mlp_rf_bayes.py
# Hybrid ensemble that follows EXACTLY the frameworks & hyperparameter spaces from:
# - RF_bayes.py
# - MLP_bayes.py
#
# It runs BayesSearchCV for RandomForestRegressor and MLPRegressor on each fold from data.final.k_fold,
# then combines predictions for each fold by simple averaging (RF + MLP) and accumulates them into y_pred / y_test.

from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Integer, Real, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# Random Forest pipeline and hyperparams (exactly as RF_bayes.py)
pipe_rf = Pipeline([
  ("scaler", StandardScaler()),
  ("rf", RandomForestRegressor())
])

hyperparams_rf = {
  "rf__n_estimators": Integer(100, 1000),
  "rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
  "rf__min_samples_split": Integer(2, 20),
  "rf__min_samples_leaf": Integer(1, 10),
  "rf__max_features": [1, "sqrt", "log2"],
  "rf__bootstrap": Categorical([True, False]),
}

# MLP pipeline and hyperparams (exactly as MLP_bayes.py)
pipe_mlp = Pipeline([
  ("scaler", StandardScaler()),
  ("mlp", MLPRegressor())
])

hyperparams_mlp = {
  "mlp__hidden_layer_sizes": Integer(100, 500),
  "mlp__solver": Categorical(["adam", "sgd"]),
  "mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
  "mlp__max_iter": Integer(150, 500),
}

# Iterate over folds (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  # RF BayesSearchCV (settings exactly from RF_bayes.py)
  bayes_rf = BayesSearchCV(
    pipe_rf,
    hyperparams_rf,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
    verbose = 4
  )
  bayes_rf.fit(x_train, y_train)

  # MLP BayesSearchCV (settings exactly from MLP_bayes.py; note n_jobs=1 as specified)
  bayes_mlp = BayesSearchCV(
    pipe_mlp,
    hyperparams_mlp,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,   # kept exactly as in MLP_bayes.py
    verbose = 4
  )
  bayes_mlp.fit(x_train, y_train)

  # Predict on this fold's test set with both models
  pred_rf = bayes_rf.predict(x_test_f)
  pred_mlp = bayes_mlp.predict(x_test_f)

  # Simple hybrid: average the two predictions
  ensemble_pred = (pred_rf + pred_mlp) / 2.0

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, ensemble_pred])

# At this point y_test and y_pred contain the concatenated true and ensemble predictions across folds.
# You may compute metrics (MSE, R2, etc.) below if desired, e.g.:
# from sklearn.metrics import mean_squared_error, r2_score
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 :", r2_score(y_test, y_pred))
