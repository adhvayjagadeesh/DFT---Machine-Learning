import numpy as np
from sklearn import clone
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

class WeightedRegressor:
  def __init__(self, estimators):
    self.estimators = estimators

  def fit(self, x, y):
    self.fitted_estimators = []
    for name, model in self.estimators:
      self.fitted_estimators.append((name, clone(model).fit(x, y)))

  def predict_stacked(self, x):
    return np.column_stack([
      model.predict(x) for _, model in self.fitted_estimators
    ])
  
  def predict(self, x, weights = None):
    if weights is None:
      weights = [1 / len(self.estimators)] * len(self.estimators)
    return np.average(self.predict_stacked(x), 1, weights)


def derive_optimal_weights(weighted_regressor: WeightedRegressor, x_train_w, y_train_w):
  model_cnt = len(weighted_regressor.estimators)
  preds = weighted_regressor.predict_stacked(x_train_w)
  def loss_fn(w):
    return mean_squared_error(y_train_w, preds @ w)
  res = minimize(loss_fn, [1 / model_cnt] * model_cnt,
    bounds = [(0, 1)] * model_cnt,
    constraints = {
      'type': 'eq',
      'fun': lambda w: np.sum(w) - 1
    }
  )
  return res.x