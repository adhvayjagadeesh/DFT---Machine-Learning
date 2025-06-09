import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# Load and clean the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]','Band gap (HSE06) [eV]', 'Direct band gap (HSE06) [eV]'])

# Define features and target
target = 'Band gap (PBE) [eV]'
X = df.drop(columns=[target])
y = df[target]

# Convert to numeric (if needed)
X = X.select_dtypes(include=[np.number])

# K-Fold CV settings
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstraps = 30  # bootstraps per fold

# Containers for results
all_y_true = []
all_y_pred = []
mae_list = []
r2_list = []

# Cross-validation with bootstrapping
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fold_preds = []

    for b in range(n_bootstraps):
        # Sample with replacement from training set
        boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[boot_idx]
        y_boot = y_train.iloc[boot_idx]

        # Train and predict
        rf = RandomForestRegressor(n_estimators=100, random_state=b)
        rf.fit(X_boot, y_boot)
        y_pred = rf.predict(X_test)
        fold_preds.append(y_pred)

    # Average predictions over bootstraps
    y_pred_avg = np.mean(fold_preds, axis=0)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred_avg)
    r2 = r2_score(y_test, y_pred_avg)

    mae_list.append(mae)
    r2_list.append(r2)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred_avg)

    print(f"Fold {fold}: MAE = {mae:.4f}, R² = {r2:.4f}")

# Overall performance
print(f"\nOverall MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Overall R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# Error calculation
errors = np.array(all_y_true) - np.array(all_y_pred)

# --- Plot 1: Error Distribution ---
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (K-Fold + Bootstrapping)')
plt.xlabel('Error (Actual - Predicted Band Gap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# --- Plot 2: Regression Plot ---
plt.figure(figsize=(8, 6))
sns.regplot(x=all_y_true, y=all_y_pred, line_kws={"color": "red"}, scatter_kws={"alpha": 0.5})
plt.xlabel('Actual Band Gap (PBE) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('Regression Plot: Predicted vs Actual Band Gap (K-Fold + Bootstrapping)')
plt.grid(True)
plt.show()
