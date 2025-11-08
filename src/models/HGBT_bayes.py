from sklearn.ensemble import HistGradientBoostingRegressor
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("hgbt", HistGradientBoostingRegressor())
])

hyperparams = {
  "hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "hgbt__max_iter": Integer(150, 800),
  "hgbt__max_leaf_nodes": Integer(20, 50),
  "hgbt__min_samples_leaf": Integer(10, 40),
  "hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
  "hgbt__max_bins": Integer(127, 255),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hgbt = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
  )
  bayes_hgbt.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hgbt.predict(x_test_f)])