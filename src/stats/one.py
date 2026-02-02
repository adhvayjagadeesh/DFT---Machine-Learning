from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
  auc,
  mean_absolute_error,
  r2_score,
  root_mean_squared_error,
)
from sklearn.model_selection import LearningCurveDisplay

from data.load import x, y
from model.create import parse_name
from model.impl import run_model
from stats.save import save_fig, save_tbl

metric_names = (
  "Name",
  "R²",
  "Adjusted R²",
  "MAE (eV)",
  "RMSE (eV)",
  "Spearman",
  "Fit time",
)


# We want this file to run as a program on its own AND be runnable by stats.many also
def run_visualize(name, output):
  y_pred, fit_time, learning_curve, feat_importances = run_model(
    *parse_name(name)
  )

  n = len(y)
  feats = list(x.columns)
  n_feat = len(feats)

  # Metrics
  r2 = r2_score(y, y_pred)
  adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_feat - 1)
  mae = mean_absolute_error(y, y_pred)
  rmse = root_mean_squared_error(y, y_pred)
  spearman = spearmanr(y, y_pred).statistic

  # Setup plots
  rows = 2
  cols = 2
  _, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

  # Plot 1: Predicted vs actual
  ax = axes[0][0]
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

  # Plot 2: REC curve
  abs_errors = np.abs(y_pred - y)
  n_steps = 100

  # y-axis ends at 2eV for now
  tolerances = np.linspace(0, 2, n_steps)
  accuracies = [
    np.mean(abs_errors <= tolerance * abs_errors.max())
    for tolerance in tolerances
  ]
  ax = axes[0][1]
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

  # Plot 3: Learning curve
  ax = axes[1][0]
  LearningCurveDisplay(**learning_curve, score_name="RMSE (eV)").plot(
    ax, line_kw={"marker": "o"}
  )
  ax.set_title("Learning curve")
  ax.set_xlabel("Training set size")
  ax.grid(True)

  # Plot 4: Permutation feature importance
  feat_range = range(n_feat)

  ax = axes[1][1]
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
    feat_range,
    mean_importances,
    yerr=abs_err_importances,
  )
  ax.set_xticks(
    feat_range, feats, rotation=30, rotation_mode="anchor", ha="right"
  )
  ax.set_ylabel("ΔRMSE (eV)")
  ax.set_title("Permutation feature importance")
  ax.grid(axis="y")

  plt.tight_layout()
  save_fig(output, name, plt)
  return (
    name,
    f"{r2:.4f}",
    f"{adj_r2:.4f}",
    f"{mae:.4f}",
    f"{rmse:.4f}",
    f"{spearman:.4f}",
    fit_time,
  )


# If ran from the CLI (not by stats.multiple)
if __name__ == "__main__":
  parser = ArgumentParser(
    "python -m stats.one", description="Visual () and metric for 1 model"
  )
  parser.add_argument("name")
  parser.add_argument(
    "-o", "--output", help="Folder to save output instead of showing it"
  )
  args = parser.parse_args()
  output = args.output
  name = args.name
  save_tbl(output, name, [run_visualize(name, output)], metric_names)
