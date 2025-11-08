from skopt import BayesSearchCV
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("", VotingRegressor([
    ("rf", RandomForestRegressor()),
    ("hgbt", HistGradientBoostingRegressor())
  ]))
])

hyperparams = {
  "__rf__n_estimators": Integer(100, 1000),
  "__rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
  "__rf__min_samples_split": Integer(2, 20),
  "__rf__min_samples_leaf": Integer(1, 10),
  "__rf__max_features": [1, "sqrt", "log2"],
  "__rf__bootstrap": Categorical([True, False]),

  "__hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "__hgbt__max_iter": Integer(150, 800),
  "__hgbt__max_leaf_nodes": Integer(20, 50),
  "__hgbt__min_samples_leaf": Integer(10, 40),
  "__hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
  "__hgbt__max_bins": Integer(127, 255),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_hybrid = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = -1,
    verbose = 4
  )
  bayes_hybrid.fit(x_train, y_train)

  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
