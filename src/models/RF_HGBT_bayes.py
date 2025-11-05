from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe_rf = Pipeline([
  ("scaler", StandardScaler()),
  ("rf", RandomForestRegressor())
])

pipe_hgbt = Pipeline([
  ("scaler", StandardScaler()),
  ("hgbt", HistGradientBoostingRegressor())
])

# RandomForest hyperparameters (unchanged)
hyperparams_rf = {
  "rf__n_estimators": Integer(100, 1000),
  "rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
  "rf__min_samples_split": Integer(2, 20),
  "rf__min_samples_leaf": Integer(1, 10),
  "rf__max_features": [1, "sqrt", "log2"],
  "rf__bootstrap": Categorical([True, False]),
}

# HistGradientBoosting hyperparameters (unchanged)
hyperparams_hgbt = {
  "hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "hgbt__max_iter": Integer(150, 800),
  "hgbt__max_leaf_nodes": Integer(20, 50),
  "hgbt__min_samples_leaf": Integer(10, 40),
  "hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
  "hgbt__max_bins": Integer(127, 255),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  # Tune RF (same as your RF_bayes.py)
  bayes_rf = BayesSearchCV(
    pipe_rf,
    hyperparams_rf,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
    verbose = 4
  )
  bayes_rf.fit(x_train, y_train)

  # Tune HGBT (same style as your HGBT_bayes.py)
  bayes_hgbt = BayesSearchCV(
    pipe_hgbt,
    hyperparams_hgbt,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,
    verbose = 4
  )
  bayes_hgbt.fit(x_train, y_train)

  # Predict and evaluate (simple average blend of the two models)
  y_test = np.concatenate([y_test, y_test_f])
  preds_rf = bayes_rf.predict(x_test_f)
  preds_hgbt = bayes_hgbt.predict(x_test_f)
  preds_blend = (preds_rf + preds_hgbt) / 2.0
  y_pred = np.concatenate([y_pred, preds_blend])
