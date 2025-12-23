from numpy import empty_like
from sklearn.model_selection import cross_val_predict

from data.final import k_fold, kf, x, y
from model.create import create_model
from model.optimize import optmize_weights, tune


# K-fold prediction (alias for cross_val_predict)
def k(names):
  return cross_val_predict(create_model(names), x, y, cv=kf)


# Tuning
def t(names):
  model = create_model(names)
  y_pred = empty_like(y)

  for x_train, y_train, x_test, indices in k_fold():
    tune(model, x_train, y_train)
    model.fit(x_train, y_train)
    y_pred[indices] = model.predict(x_test)
  return y_pred


# Weighting
def w(names):
  model = create_model(names)
  y_pred = empty_like(y)

  for x_train, y_train, x_test, indices in k_fold():
    optmize_weights(model, x_train, y_train)
    model.fit(x_train, y_train)
    y_pred[indices] = model.predict(x_test)

  return y_pred


# Weighting and tuning
def wt(names):
  model = create_model(names)
  y_pred = empty_like(y)

  for x_train, y_train, x_test, indices in k_fold():
    tune(model, x_train, y_train)
    optmize_weights(model, x_train, y_train)
    model.fit(x_train, y_train)
    y_pred[indices] = model.predict(x_test)
