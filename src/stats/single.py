import argparse
import importlib
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, auc
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
from data.final import feat_cnt

parser = argparse.ArgumentParser(prog = "1-model stat", description = "Prediction and error for 1 model")
parser.add_argument("model")
parser.add_argument("-s", "--save", help = "Save the figure instead of showing it")
modelName = parser.parse_args().model
save_loc = parser.parse_args().save
model = importlib.import_module(f"models.{modelName}")

for var in ["y_test", "y_pred"]:
  if not hasattr(model, var):
    print(f"Model '{modelName}' is missing required variable: {var}")
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
    f"{modelName} summary:\n\n"
    f"R²:          {r2:.4f}\n"
    f"Adjusted R²: {adj_r2:.4f}\n"
    f"MAE:         {mae:.4f} eV\n"
    f"RMSE:        {rmse:.4f} eV\n"
    f"Spearman:    {spearman:.4f}"
)

# Plotting
row = 2
col = 2
fig, axes = plt.subplots(row, col, figsize=(row * 4, col * 4))

# "Plot" 1: Performance summary
axes[0][0].axis('off')  # Hide the axes
axes[0][0].text(0.25, 0.5, perf_summary, fontsize=12, ha='left', va='center', family='monospace')

# Plot 2: Prediction vs Actual
axes[0][1].scatter(y_test, y_pred, color='purple', alpha=0.7, label="Prediction")
axes[0][1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label = "Ideal")
axes[0][1].set_xlabel("Actual Band Gap [eV]")
axes[0][1].set_ylabel("Predicted Band Gap [eV]")
axes[0][1].set_title("Predicted vs Actual Band Gap")
axes[0][1].legend()
axes[0][1].grid(True)

# Plot 3: Error Distribution
errors = y_pred - y_test
axes[1][0].hist(errors, bins = 50, color = "teal", alpha = 0.7, edgecolor = "black")
axes[1][0].set_xlabel("Prediction Error (eV)")
axes[1][0].set_ylabel("Frequency")
axes[1][0].set_title("Prediction Error Distribution")
axes[1][0].grid(True)

# Plot 4: REC curve
abs_errors = np.abs(errors)
tolerances = np.linspace(0, 1, 100)
accuracies = []
for tolerance in tolerances:
  accuracies.append(np.mean(abs_errors <= tolerance * abs_errors.max()))
area_over = 1- auc(tolerances, accuracies)
axes[1][1].plot(tolerances, accuracies, label=f"AOC = {area_over: .4f}",)
axes[1][1].set_xlabel("Tolerance (eV)")
axes[1][1].set_ylabel("Accuracy (%)")
axes[1][1].set_title("Prediction tolerance vs Accuracy")
axes[1][1].legend(loc = 4)
axes[1][1].grid(True)

plt.tight_layout()
if save_loc:
    plt.savefig(f"{save_loc}/{modelName}.svg")
else:
  plt.show()