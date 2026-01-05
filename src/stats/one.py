from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import get_backend, rcParams
from scipy.stats import spearmanr
from sklearn.metrics import (
  auc,
  mean_absolute_error,
  r2_score,
  root_mean_squared_error,
)
from sklearn.model_selection import LearningCurveDisplay

from data.load import feat_indices, x, y
from model.create import parse_name
from model.impl import run_model

backend = get_backend()

# Increase default font size by a bit
rcParams["font.size"] = 11

metric_names = (
  "R²",
  "Adjusted R²",
  "MAE",
  "RMSE",
  "Spearman",
  "Fit time",
)


# We want this file to run as a program on its own AND be runnable by stats.many also
def run_visualize(name, save_loc):
  y_pred, fit_time, learning_curve, feat_importances = run_model(
    *parse_name(name)
  )

  n = len(y)
  n_feat = x.shape[1]

  # Metrics
  r2 = r2_score(y, y_pred)
  adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_feat - 1)
  mae = mean_absolute_error(y, y_pred)
  rmse = root_mean_squared_error(y, y_pred)
  spearman = spearmanr(y, y_pred).statistic

  # Setup plots
  rows = 2
  cols = 3
  _, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

  # "Plot" 1: Performance summary
  ax = axes[0][0]
  metrics = (
    f"{r2:.4f}",
    f"{adj_r2:.4f}",
    f"{mae:.4f} eV",
    f"{rmse:.4f} eV",
    f"{spearman:.4f}",
    fit_time,
  )
  tbl = ax.table([*zip(metric_names, metrics)], loc="center", cellLoc="left")
  tbl.scale(1, 2)
  for _, cell in tbl.get_celld().items():
    cell.set_linewidth(0.4)
  ax.axis("off")
  ax.axis("tight")
  ax.set_title(f"{name} summary")

  # Plot 2: Predicted vs actual
  ax = axes[0][1]
  ax.scatter(y, y_pred, label="Prediction", alpha=0.5, color="purple")
  ax.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--",
    label="Ideal",
  )
  ax.set_xlabel("Actual Band Gap (eV)")
  ax.set_ylabel("Predicted Band Gap (eV)")
  ax.set_title("Predicted vs Actual Band Gap")
  ax.legend()
  ax.grid(True)

  # Plot 3: Error distribution
  errors = y_pred - y
  ax = axes[0][2]
  ax.hist(errors, bins=40, color="turquoise", edgecolor="black")
  ax.set_xlabel("Prediction Error (eV)")
  ax.set_ylabel("Frequency")
  ax.set_title("Prediction error distribution")
  ax.grid(True)

  # Plot 4: REC curve
  abs_errors = np.abs(errors)
  n_steps = 100

  # y-axis ends at 2eV for now
  tolerances = np.linspace(0, 2, n_steps)
  accuracies = [
    np.mean(abs_errors <= tolerance * abs_errors.max())
    for tolerance in tolerances
  ]
  ax = axes[1][0]
  ax.plot(
    tolerances,
    accuracies,
    label=f"AUC = {auc(np.linspace(0, 1, n_steps), accuracies):.4f}",
  )
  ax.set_xlabel("Tolerance (eV)")
  ax.set_ylabel("Accuracy")
  ax.set_title("Prediction accuracy vs tolerance")
  ax.legend()
  ax.grid(True)

  # Plot 5: Learning curve
  ax = axes[1][1]
  LearningCurveDisplay(**learning_curve, score_name="RMSE (eV)").plot(
    ax, line_kw={"marker": "o"}
  )
  ax.set_title("Learning curve")
  ax.set_xlabel("Training set size")
  ax.grid(True)

  # Plot 6: Permutation feature importance
  ax = axes[1][2]
  mean_importances = np.mean(feat_importances, 0)
  abs_err_importances = np.abs(
    np.stack(
      [
        mean_importances - np.min(feat_importances, 0),
        mean_importances - np.max(feat_importances, 0),
      ]
    )
  )
  ax.bar(
    feat_indices,
    mean_importances,
    yerr=abs_err_importances,
  )
  ax.set_xlabel("Feature #")
  ax.set_ylabel("ΔRMSE (eV)")
  ax.set_title("Permutation feature importance")
  ax.grid(axis="y")

  plt.tight_layout()
  if save_loc:
    plt.savefig(f"{save_loc}/{name}.svg")
  elif backend != "agg":
    plt.show()
  else:
    # Failsafe for "UserWarning: FigureCanvasAgg is non-interactive, and thus cannot
    # be shown" to prevent forgetting graphics backend and losing 67 hours of progress.
    print(
      "IMMEDIATELY install and set the Matplotlib interactive backend .\n",
      "I hereby rescue you this time and this time only. INSTALL IT NOW.\n",
      f"Figured saved as {name}.svg",
    )
    plt.savefig(f"./{name}.svg")

  return name, *metrics


# If ran from the CLI (not by stats.multiple)
if __name__ == "__main__":
  parser = ArgumentParser(
    "python -m stats.one", description="Visualization and stats for a model"
  )
  parser.add_argument("name")
  parser.add_argument(
    "-s", "--save", help="Save the figure instead of showing it"
  )
  args = parser.parse_args()
  run_visualize(args.name, args.save)
