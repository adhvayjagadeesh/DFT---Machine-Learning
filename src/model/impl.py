from time import perf_counter_ns

from numpy import empty, empty_like, linspace
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error

from data.load import k, k_fold, x, y
from model.create import create_model
from model.optimize import optmize_weights, tune


def run_model(names, mode):
  model = create_model(names)
  y_pred = empty_like(y)

  # Feature importance data (ΔRMSE)
  feat_importances = empty((k, x.shape[1]))

  # Number of points on learning curve
  n_points = 5

  # Fractions of data to use for learning curve
  fracs = linspace(0.2, 1, n_points)
  n_row_all = len(x)
  fit_time = 0

  # Learning curve data
  learning_curve = {
    "train_sizes": empty(n_points),
    "test_scores": empty((n_points, k)),
    "train_scores": empty((n_points, k)),
  }

  # Learning curve loop
  for i, frac in enumerate(fracs):
    n_row = int(n_row_all * frac)

    # Simple slicing to save memory, allowed because CSV is shuffled (x is sliced too in k_fold)
    y_slc = y[0:n_row]

    # CV loop
    for j, (x_train, y_train, x_test, indices) in enumerate(
      k_fold(x[0:n_row], y_slc)
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

      # Add learning curve data
      learning_curve["test_scores"][i][j] = root_mean_squared_error(
        y_test, y_pred_f
      )
      learning_curve["train_scores"][i][j] = root_mean_squared_error(
        y_train, model.predict(x_train)
      )

      # Save to y_pred and get feature importance only for 100%-data run
      if frac == 1:
        y_pred[indices] = y_pred_f
        feat_importances[j] = permutation_importance(
          model, x_test, y_test, scoring="neg_root_mean_squared_error"
        ).importances_mean

    # Approximate training set size so we don't touch k_fold (KFold tries to balance folds as perfectly as possible already)
    learning_curve["train_sizes"][i] = n_row * (1 - 1 / k)

  # Convert fit time to seconds and average it across folds
  fit_time /= 10**9 * k
  return (
    y_pred,
    # HH:MM:SS.SSS
    f"{int(fit_time // 3600):02}:{int(fit_time % 3600 // 60):02}:{fit_time % 60:06.3f}",
    learning_curve,
    feat_importances,
  )
