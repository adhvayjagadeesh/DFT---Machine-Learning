from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import get_backend
from scipy.stats import spearmanr
from sklearn.metrics import (
  auc,
  mean_absolute_error,
  r2_score,
  root_mean_squared_error,
)

from data.final import x, y
from model.create import Model
from model.impl import run_model

backend = get_backend()
possible_modes = ["k", "w", "t", "wt"]
possible_names = list(Model.__members__)


def run_visualize(mode, names, save_loc):
  y_pred, fit_time, learning_curve = run_model(mode, names)

  n = len(y)
  n_feat = x.shape[1]
  name = f"{'+'.join(names)}_{mode}"

  # Metrics
  r2 = r2_score(y, y_pred)
  adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_feat - 1)
  mae = mean_absolute_error(y, y_pred)
  rmse = root_mean_squared_error(y, y_pred)
  spearman, _ = spearmanr(y, y_pred)
  perf_summary = (
    f"{name} summary:\n\n"
    f"R²:          {r2:.4f}\n"
    f"Adjusted R²: {adj_r2:.4f}\n"
    f"MAE:         {mae:.4f} eV\n"
    f"RMSE:        {rmse:.4f} eV\n"
    f"Spearman:    {spearman:.4f}\n"
    f"Fit time:    {fit_time}"
  )

  # Plotting
  rows = 2
  cols = 3
  _, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

  # "Plot" 1: Performance summary
  ax = axes[0][0]
  ax.axis("off")  # Hide the axes
  ax.text(
    0.25,
    0.5,
    perf_summary,
    fontsize=12,
    ha="left",
    va="center",
    family="monospace",
  )

  # Plot 2: Prediction vs Actual
  ax = axes[0][1]
  ax.scatter(y, y_pred, color="purple", alpha=0.7, label="Prediction")
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

  # Plot 3: Error Distribution
  errors = y_pred - y
  ax = axes[0][2]
  ax.hist(errors, bins=50, color="teal", alpha=0.7, edgecolor="black")
  ax.set_xlabel("Prediction Error (eV)")
  ax.set_ylabel("Frequency")
  ax.set_title("Prediction Error Distribution")
  ax.grid(True)

  # Plot 4: REC curve
  abs_errors = np.abs(errors)
  n_steps = 100

  # y-axis end at 2eV for now
  tolerances = np.linspace(0, 2, n_steps)
  accuracies = [np.mean(abs_errors <= i * abs_errors.max()) for i in tolerances]
  ax = axes[1][0]
  ax.plot(
    tolerances,
    accuracies,
    label=f"AUC = {auc(np.linspace(0, 1, n_steps), accuracies): .4f}",
  )
  ax.set_xlabel("Tolerance (eV)")
  ax.set_ylabel("Accuracy (%)")
  ax.set_title("Prediction accuracy vs tolerance")
  ax.legend(loc="best")
  ax.grid(True)

  # Plot 5: Learning curve
  ax = axes[1][1]
  ax.plot(
    learning_curve["sizes"],
    learning_curve["train_scores"],
    label="Train scores",
  )
  ax.plot(
    learning_curve["sizes"], learning_curve["cv_scores"], label="CV scores"
  )
  ax.set_xlabel("Sample size")
  ax.set_ylabel("RMSE (eV)")
  ax.set_title("Learning curve")
  ax.legend(loc="best")

  # Plot 6: Feature importance
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

  return name, r2, adj_r2, mae, rmse, spearman, fit_time


# If ran from the CLI (not by stats.multiple)
if __name__ == "__main__":
  parser = ArgumentParser(
    "1-model stat", description="Visualization and stats for a model"
  )
  parser.add_argument("mode", choices=possible_modes)
  parser.add_argument("names", choices=possible_names, nargs="+")
  parser.add_argument(
    "-s", "--save", help="Save the figure instead of showing it"
  )
  args = parser.parse_args()
  run_visualize(args.mode, args.names, args.save)
