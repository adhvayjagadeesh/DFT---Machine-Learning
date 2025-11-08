from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
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
pipe = Pipeline([
  ("scaler", StandardScaler()),
  VotingRegressor([
    ("rf", RandomForestRegressor()),
    ("mlp", MLPRegressor())
  ])
])

hyperparams = {
  "__rf__n_estimators": Integer(100, 1000),
  "__rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
  "__rf__min_samples_split": Integer(2, 20),
  "__rf__min_samples_leaf": Integer(1, 10),
  "__rf__max_features": [1, "sqrt", "log2"],
  "__rf__bootstrap": Categorical([True, False]),
  
  "__mlp__hidden_layer_sizes": Integer(100, 500),
  "__mlp__solver": Categorical(["adam", "sgd"]),
  "__mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
  "__mlp__max_iter": Integer(150, 500),
}

# Iterate over folds (no scaling here because pipelines handle scaling)
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1,
  )
  bayes_hybrid .fit(x_train, y_train)

  # Combine fold results into global arrays
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
