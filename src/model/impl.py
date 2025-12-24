from time import perf_counter_ns
from typing import Any

from numpy import empty_like, linspace
from sklearn.metrics import root_mean_squared_error

from data.final import k, k_fold, x, y
from model.create import create_model
from model.optimize import optmize_weights, tune


def run_model(mode, names):
  # Weighting is only for ensemble
  if len(names) < 2:
    assert "w" not in mode
  model = create_model(names)
  y_pred = empty_like(y)

  # Learning curve data
  learning_curve = {"sizes": [], "cv_scores": [], "train_scores": []}

  # Fractions of data to use for learning curve
  fracs = linspace(0.2, 1, 5)
  n_row_all = len(x)
  fit_time = 0

  # Run the model on those data fractions
  for i in fracs:
    n_row = int(n_row_all * i)
    rmse_train = 0
    rmse_test = 0

    # Slice to reuse and save memory
    x_slc = x[0:n_row]
    y_slc: Any = y[0:n_row]  # Any type to quiet the linter

    for x_train, y_train, x_test, indices in k_fold(x_slc, y_slc):
      start_ns = perf_counter_ns()
      if "t" in mode:
        tune(model, x_train, y_train)
      if "w" in mode:
        optmize_weights(model, x_train, y_train)
      model.fit(x_train, y_train)

      # Time only for full-data run
      if i == 1:
        fit_time += perf_counter_ns() - start_ns

      # By _f I mean fold
      y_pred_f = model.predict(x_test)

      # Save y_pred only for full-data run
      if i == 1:
        y_pred[indices] = y_pred_f
      rmse_test += root_mean_squared_error(y_slc.iloc[indices], y_pred_f)
      rmse_train += root_mean_squared_error(y_train, model.predict(x_train))

    learning_curve["sizes"].append(n_row)
    learning_curve["train_scores"].append(rmse_train / k)
    learning_curve["cv_scores"].append(rmse_test / k)

  # Convert fit time to seconds and average it across folds
  fit_time /= 10**9 * k
  return (
    y_pred,
    f"{int(fit_time // 3600):02}:{int(fit_time % 3600 // 60):02}:{int(fit_time % 60):02}",
    learning_curve,
  )
