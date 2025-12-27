from numpy import empty_like, sum
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from data.prepare import k_fold, kf
from model.create import Model


# Get the hyperparameters for 1 model
def _get_hyperparams(model: Model):
  if model == Model.gbt:
    hyperparams = {
      "n_estimators": Integer(200, 800),
      "learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
      "max_depth": Integer(3, 10),
      "min_samples_split": Integer(2, 15),
      "min_samples_leaf": Integer(3, 10),
      "subsample": Real(0.7, 1),
      "max_features": Categorical([None, "sqrt", "log2"]),
    }
  elif model == Model.hgbt:
    hyperparams = {
      "learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
      "max_iter": Integer(150, 800),
      "max_leaf_nodes": Integer(20, 50),
      "min_samples_leaf": Integer(10, 40),
      "l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
      "max_bins": Integer(127, 255),
    }
  elif model == Model.mlp:
    hyperparams = {
      "hidden_layer_sizes": Integer(100, 500),
      "solver": Categorical(["adam", "sgd"]),
      "learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
      "max_iter": Integer(150, 500),
    }
  elif model == Model.rf:
    hyperparams = {
      "n_estimators": Integer(100, 1000),
      "max_depth": Integer(1, 75),
      "min_samples_split": Integer(2, 20),
      "min_samples_leaf": Integer(3, 10),
      "max_features": Categorical(["sqrt", "log2"]),
      "bootstrap": Categorical([True, False]),
    }
  elif model == Model.xgb:
    hyperparams = {
      "max_depth": Integer(1, 75),
      "min_child_weight": Integer(1, 8),
      "eta": Real(1e-2, 0.2, prior="log-uniform"),
      "n_estimators": Integer(100, 800),
      "subsample": Real(0.7, 1.0),
      "colsample_bytree": Real(0.6, 1.0),
      "reg_alpha": Real(1e-3, 2.0, prior="log-uniform"),
      "reg_lambda": Real(1e-3, 2.0, prior="log-uniform"),
      "gamma": Real(1e-3, 2.0, prior="log-uniform"),
      "tree_method": Categorical(["hist", "approx"]),
    }
  elif model == Model.abdt:
    hyperparams = {
      # AdaBoost
      "n_estimators": Integer(50, 500),
      "learning_rate": Real(1e-2, 1.0, prior="log-uniform"),
      "loss": Categorical(["linear", "square", "exponential"]),
      # DecisionTreeRegressor
      "estimator__max_depth": Integer(1, 75),
      "estimator__min_samples_split": Integer(2, 20),
      "estimator__min_samples_leaf": Integer(3, 10),
      "estimator__max_features": Categorical(["sqrt", "log2"]),
    }
  elif model == Model.abet:
    hyperparams = {
      # AdaBoost Parameters
      "n_estimators": Integer(50, 500),
      "learning_rate": Real(1e-2, 1.0, prior="log-uniform"),
      "loss": Categorical(["linear", "square", "exponential"]),
      # ExtraTreeRegressor (similar to Decision tree)
      "estimator__splitter": Categorical(["random", "best"]),
      "estimator__max_depth": Integer(1, 75),
      "estimator__min_samples_split": Integer(2, 20),
      "estimator__min_samples_leaf": Integer(3, 10),
      "estimator__max_features": Categorical(["sqrt", "log2"]),
    }
  elif model == Model.ets:
    hyperparams = {
      "n_estimators": Integer(100, 1000),
      "max_depth": Integer(1, 75),
      "min_samples_split": Integer(2, 20),
      "min_samples_leaf": Integer(3, 10),
      "max_features": Categorical(["sqrt", "log2"]),
      "bootstrap": Categorical([True, False]),
    }
  return hyperparams


# Optimize weights
def optmize_weights(model, x_train, y_train):
  # Assume model here means a 1+ model Pipeline so model[1] will be a VotingRegressor (because we always assert in w/wb)
  n_model = len(model[1].estimators)

  # Unweighted CV predictions (could've used cross_val_predict if it allows method="transform")
  y_pred = empty_like(y_train, shape=(y_train.shape[0], n_model))
  for x_train_w, y_train_w, x_test_w, indices in k_fold(x_train, y_train):
    model.fit(x_train_w, y_train_w)
    y_pred[indices] = model.transform(x_test_w)

  def loss_fn(w):
    return mean_squared_error(y_train, y_pred @ w)

  res = minimize(
    loss_fn,
    [1 / n_model] * n_model,
    bounds=[(0, 1)] * n_model,
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1},
  )
  model.set_params(__weights=res.x)


def tune(model, x_train, y_train):
  hyperparams = {}
  for name, submodel in model[1].estimators:
    bayes = BayesSearchCV(
      submodel,
      _get_hyperparams(Model[name]),
      cv=kf,
      n_iter=25,
      n_jobs=-1,
      pre_dispatch="1.5*n_jobs",
      refit=False,
    )
    bayes.fit(x_train, y_train)
    prefix = f"__{name}__"
    hyperparams.update({prefix + k: v for k, v in bayes.best_params_.items()})
  model.set_params(**hyperparams)
