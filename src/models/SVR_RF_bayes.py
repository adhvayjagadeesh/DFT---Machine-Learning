# hybrid_bayes.py
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Integer, Real, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# Random Forest pipeline + hyperparams (exactly as in RF_bayes.py)
pipe_rf = Pipeline([
  ("scaler", StandardScaler()),
  VotingRegressor([
    ("svr", SVR(max_iter = 1000000)),
    ("rf", RandomForestRegressor())
  ])
])

hyperparams = {
  "__rf__n_estimators": Integer(100, 1000),
  "__rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
  "__rf__min_samples_split": Integer(2, 20),
  "__rf__min_samples_leaf": Integer(1, 10),
  "__rf__max_features": [1, "sqrt", "log2"],
  "__rf__bootstrap": Categorical([True, False]),
  
  "__svr__C": Real(1e-3, 1e+6, prior="log-uniform"),
  "__svr__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
  "__svr__degree": Integer(1, 9),
  "__svr__epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
  "__svr__kernel": Categorical(["linear", "poly", "rbf"]),
}

# Iterate over the same k_fold generator (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe_rf,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
  )
  bayes_hybrid.fit(x_train, y_train)

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])