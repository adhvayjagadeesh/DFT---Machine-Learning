import importlib
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import get_backend
from scipy.stats import spearmanr
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, r2_score

from data.final import feat_cnt

backend = get_backend()


def run_model(name, save_loc):
  model = importlib.import_module(f"models.{name}")
  for var in ["y_test", "y_pred"]:
    if not hasattr(model, var):
      print(f"Model '{name}' is missing required variable: {var}")
      sys.exit(1)
  y_test = model.y_test
  y_pred = model.y_pred

  # Metrics
  n = len(y_test)
  r2 = r2_score(y_test, y_pred)
  adj_r2 = 1 - (1 - r2) * (n - 1) / (n - feat_cnt - 1)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  spearman, _ = spearmanr(y_test, y_pred)

  perf_summary = (
    f"{name} summary:\n\n"
    f"R²:          {r2:.4f}\n"
    f"Adjusted R²: {adj_r2:.4f}\n"
    f"MAE:         {mae:.4f} eV\n"
    f"RMSE:        {rmse:.4f} eV\n"
    f"Spearman:    {spearman:.4f}"
  )

  # Plotting
  rows = 2
  cols = 2
  _, axes = plt.subplots(rows, cols, figsize=(rows * 4, cols * 4))

  # "Plot" 1: Performance summary
  ax = axes[0][0]
  ax.axis("off")  # Hide the axes
  ax.text(
    0.25, 0.5, perf_summary, fontsize=12, ha="left", va="center", family="monospace"
  )

  # Plot 2: Prediction vs Actual
  ax = axes[0][1]
  ax.scatter(y_test, y_pred, color="purple", alpha=0.7, label="Prediction")
  ax.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Ideal"
  )
  ax.set_xlabel("Actual Band Gap [eV]")
  ax.set_ylabel("Predicted Band Gap [eV]")
  ax.set_title("Predicted vs Actual Band Gap")
  ax.legend()
  ax.grid(True)

  # Plot 3: Error Distribution
  errors = y_pred - y_test
  ax = axes[1][0]
  ax.hist(errors, bins=50, color="teal", alpha=0.7, edgecolor="black")
  ax.set_xlabel("Prediction Error (eV)")
  ax.set_ylabel("Frequency")
  ax.set_title("Prediction Error Distribution")
  ax.grid(True)

  # Plot 4: REC curve
  abs_errors = np.abs(errors)
  tolerances = np.linspace(0, 1, 100)
  accuracies = []
  for tolerance in tolerances:
    accuracies.append(np.mean(abs_errors <= tolerance * abs_errors.max()))
  area_over = 1 - auc(tolerances, accuracies)
  ax = axes[1][1]
  ax.plot(
    tolerances,
    accuracies,
    label=f"AOC = {area_over: .4f}",
  )
  ax.set_xlabel("Tolerance (eV)")
  ax.set_ylabel("Accuracy (%)")
  ax.set_title("Prediction tolerance vs Accuracy")
  ax.legend(loc=4)
  ax.grid(True)

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

  return name, r2, adj_r2, mae, rmse, spearman


# If ran from the CLI (not by stats.multiple)
if __name__ == "__main__":
  parser = ArgumentParser(
    "1-model stat", description="Prediction and error for 1 model"
  )
  parser.add_argument("model")
  parser.add_argument("-s", "--save", help="Save the figure instead of showing it")
  args = parser.parse_args()
  run_model(args.model, args.save)
