import numpy as np
from scipy.optimize import minimize
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted, validate_data
from skopt.space import Categorical, Integer, Real


class WeightedRegressor(RegressorMixin, BaseEstimator):
  def __init__(self, estimators):
    self.estimators = estimators

  def fit(self, x, y):
    x, y = validate_data(self, x, y, accept_sparse=False)
    self.is_fitted_ = True
    self.fitted_estimators_ = []
    for name, model in self.estimators:
      self.fitted_estimators_.append((name, clone(model).fit(x, y)))
    return self

  def predict_stacked(self, x):
    check_is_fitted(self)
    x = validate_data(self, x, reset=False)
    return np.column_stack([model.predict(x) for _, model in self.fitted_estimators_])

  def predict(self, x, weights=None):
    check_is_fitted(self)
    if weights is None:
      weights = [1 / len(self.estimators)] * len(self.estimators)
    return np.average(self.predict_stacked(x), 1, weights)


# If this file is run directly, check the custom estimator
if __name__ == "__main__":
  check_estimator(
    WeightedRegressor(
      [("rf", RandomForestRegressor()), ("gbt", GradientBoostingRegressor())]
    )
  )


def derive_optimal_weights(weighted_regressor: WeightedRegressor, x_train_w, y_train_w):
  model_cnt = len(weighted_regressor.estimators)
  preds = weighted_regressor.predict_stacked(x_train_w)

  def loss_fn(w):
    return mean_squared_error(y_train_w, preds @ w)

  res = minimize(
    loss_fn,
    [1 / model_cnt] * model_cnt,
    bounds=[(0, 1)] * model_cnt,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
  )
  return res.x


def get_hyperparam(model):
  if model == "gbt":
    hyperparam = {
      "n_estimators": Integer(200, 800),
      "learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
      "max_depth": Integer(3, 10),
      "min_samples_split": Integer(2, 15),
      "min_samples_leaf": Integer(1, 10),
      "subsample": Real(0.7, 1),
      "max_features": Categorical(["sqrt", 0.7, None]),
    }
  elif model == "hgbt":
    hyperparam = {
      "learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
      "max_iter": Integer(150, 800),
      "max_leaf_nodes": Integer(20, 50),
      "min_samples_leaf": Integer(10, 40),
      "l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
      "max_bins": Integer(127, 255),
    }
  elif model == "mlp":
    hyperparam = {
      "hidden_layer_sizes": Integer(100, 500),
      "solver": Categorical(["adam", "sgd"]),
      "learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
      "max_iter": Integer(150, 500),
    }
  elif model == "rf":
    hyperparam = {
      "n_estimators": Integer(100, 1000),
      "max_depth": Categorical([None, 10, 25, 50, 75, 100]),
      "min_samples_split": Integer(2, 20),
      "min_samples_leaf": Integer(1, 10),
      "max_features": [1, "sqrt", "log2"],
      "bootstrap": Categorical([True, False]),
    }
  elif model == "svr":
    hyperparam = {
      "C": Real(1e-3, 1e6, prior="log-uniform"),
      "gamma": Real(1e-6, 1e1, prior="log-uniform"),
      "degree": Integer(1, 9),
      "epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
      "kernel": Categorical(["linear", "poly", "rbf"]),
    }
  elif model == "xgb":
    hyperparam = {
      "max_depth": Integer(3, 10),
      "min_child_weight": Integer(1, 8),
      "learning_rate": Real(1e-2, 0.2, prior="log-uniform"),
      "n_estimators": Integer(100, 800),
      "subsample": Real(0.7, 1.0),
      "colsample_bytree": Real(0.6, 1.0),
      "reg_alpha": Real(1e-5, 0.5, prior="log-uniform"),
      "reg_lambda": Real(0.5, 2.0, prior="log-uniform"),
      "gamma": Real(0.0, 2.0),
      "tree_method": Categorical(["hist", "approx"]),
    }
  prefix = f"{model}__"
  return {prefix + k: v for k, v in hyperparam.items()}


def get_hyperparams(models):
  hyperparams = {}
  for model in models:
    hyperparams.update(get_hyperparam(model))
  return {"__" + k: v for k, v in hyperparams.items()}
