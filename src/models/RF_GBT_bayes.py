from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

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

hyperparams = {
  "__rf__n_estimators": Integer(100, 1000),
  "__rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
  "__rf__min_samples_split": Integer(2, 20),
  "__rf__min_samples_leaf": Integer(1, 10),
  "__rf__max_features": [1, "sqrt", "log2"],
  "__rf__bootstrap": Categorical([True, False]),

  "__gbt__n_estimators": Integer(200, 800),
  "__gbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "__gbt__max_depth": Integer(3, 10),
  "__gbt__min_samples_split": Integer(2, 15),
  "__gbt__min_samples_leaf": Integer(1, 10),
  "__gbt__subsample": Real(0.7, 1),
  "__gbt__max_features": Categorical(["sqrt", 0.7, None]),
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

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_hybrid.predict(x_test_f)])
