from time import perf_counter_ns

from numpy import empty, empty_like, linspace
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error

from data.prepare import k, k_fold, x, y
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

  # Feature importance data (delta RMSE)
  feat_importances = empty((k, x.shape[1]))

  # Fractions of data to use for learning curve
  fracs = linspace(0.2, 1, 5)
  n_row_all = len(x)
  fit_time = 0

  # Learning curve loop
  for frac in fracs:
    n_row = int(n_row_all * frac)
    rmse_train = 0
    rmse_test = 0

    # Simple slicing to save memory (allowed because CSV is shuffled)
    x_slc = x[0:n_row]
    y_slc = y[0:n_row]

    # Cross-validation loop
    for i, (x_train, y_train, x_test, indices) in enumerate(
      k_fold(x_slc, y_slc)
    ):
      start_ns = perf_counter_ns()
      if "t" in mode:
        tune(model, x_train, y_train)
      if "w" in mode:
        optmize_weights(model, x_train, y_train)
      model.fit(x_train, y_train)

      # Fit time only for 100%-data run
      if frac == 1:
        fit_time += perf_counter_ns() - start_ns

      # By _f (so that it doesn't conflict with the outer y_pred) I mean fold
      y_pred_f = model.predict(x_test)
      y_test = y_slc.iloc[indices]

      # Save to y_pred and get feature importance only for 100%-data run
      if frac == 1:
        y_pred[indices] = y_pred_f
        feat_importances[i] = permutation_importance(
          model, x_test, y_test, scoring="neg_root_mean_squared_error"
        ).importances_mean
      rmse_test += root_mean_squared_error(y_test, y_pred_f)
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
    feat_importances,
  )
