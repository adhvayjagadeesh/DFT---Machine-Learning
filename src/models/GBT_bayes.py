from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
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
  ("gbt", GradientBoostingRegressor())
])

hyperparams = {
  "gbt__n_estimators": Integer(200, 800),
  "gbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
  "gbt__max_depth": Integer(3, 10),
  "gbt__min_samples_split": Integer(2, 15),
  "gbt__min_samples_leaf": Integer(1, 10),
  "gbt__subsample": Real(0.7, 1), 
  "gbt__max_features": Categorical(["sqrt", 0.7, None]), 
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_gbt = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with XGB, worsening performance
  )
  bayes_gbt.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_gbt.predict(x_test_f)])