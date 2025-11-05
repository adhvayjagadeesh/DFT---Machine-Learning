from skopt import BayesSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
from skopt.space import Real, Integer, Categorical
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("mlp", MLPRegressor())
])

hyperparams = {
  "mlp__hidden_layer_sizes": Integer(100, 500),
  "mlp__solver": Categorical(["adam", "sgd"]),
  "mlp__learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
  "mlp__max_iter": Integer(150, 500),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
  bayes_mlp = BayesSearchCV(
    pipe,
    hyperparams,
    cv = k_,
    n_iter = 20,
    n_jobs = 1, # This default to all threads, which conflict with mlp, worsening performance
    verbose = 4
  )
  bayes_mlp.fit(x_train, y_train)

  # Predict and evaluate
  y_test = np.concatenate([y_test, y_test_f])
  y_pred = np.concatenate([y_pred, bayes_mlp.predict(x_test_f)])