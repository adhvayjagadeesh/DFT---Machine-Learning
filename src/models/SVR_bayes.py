from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from data.final import k_fold, k_
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils.hybrid import make_hyperparam

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("svr", SVR(max_iter = 1000000)) # Some of the cv is taking too long
])

hyperparams = make_hyperparam("svr")

# By _f I mean fold, no scaling cuz pipeline is gonna handle that
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_svr = BayesSearchCV(pipe,
              hyperparams, 
              cv = k_,
              n_jobs = -1,
              n_iter = 20,
              verbose = 4)
  bayes_svr.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_svr.predict(x_test_f)])