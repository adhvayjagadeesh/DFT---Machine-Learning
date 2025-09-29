import argparse
import importlib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="1-model stat", description="Prediction and error for 1 model")
parser.add_argument("model")
modelName = parser.parse_args().model

model = importlib.import_module(f"models.{modelName}")
for var in ["k", "y_true", "y_pred"]:
  if not hasattr(model, var):
    print(f"Model '{args.model}' is missing required variable: {var}")
    sys.exit(1)
y_true = model.y_true
y_pred = model.y_pred
k = model.k

#y_true = [1,2,3,4]
#y_pred = [1,2,3,4]
#k = 4

# Metrics
n = len(y_true)
r2 = r2_score(y_true, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
spearman, _ = spearmanr(y_true, y_pred)

print(f"\nPerformance summary:")
print(f"R²:            {r2:.4f}")
print(f"Adjusted R²    {adj_r2:.4f}")
print(f"MAE:           {mae:.4f} eV")
print(f"RMSE:          {rmse:.4f} eV")
print(f"Spearman correlation: {spearman:.4f}")

# Plot 1: Prediction vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, color='purple', alpha=0.7, label="Prediction")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Actual Band Gap [eV]")
plt.ylabel("Predicted Band Gap [eV]")
plt.title("Predicted vs Actual Band Gap")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Error Distribution
plt.figure(figsize=(6, 4))
plt.hist(y_pred - y_true, bins=50, color='teal', alpha=0.7, edgecolor='black')
plt.xlabel("Prediction Error (Pred - Actual)")
plt.title("Prediction Error Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()