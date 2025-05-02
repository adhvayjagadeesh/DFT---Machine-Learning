import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

# Load and clean the dataset
df = pd.read_csv('/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv')
df = df.drop(columns=['Band gap (G₀W₀) [eV]', 'Direct band gap (PBE) [eV]', 'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]'])

# Define features and target
target = 'Band gap (PBE) [eV]'
X = df.drop(columns=[target])
y = df[target]

# Convert to numeric (if needed)
X = X.select_dtypes(include=[np.number])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Check if the column count matches before re-assigning column names
if X_imputed.shape[1] == X.shape[1]:
    X = pd.DataFrame(X_imputed, columns=X.columns)
else:
    print(f"Warning: Column count mismatch. Original: {X.shape[1]}, After Imputation: {X_imputed.shape[1]}")
    X = pd.DataFrame(X_imputed)  # Assign without column names to avoid ValueError

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bootstrapping (100 iterations)
n_iterations = 100
mae_scores_bootstrap = []
r2_scores_bootstrap = []
errors_bootstrap = []

# K-Fold Cross-Validation (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores_kfold = []
r2_scores_kfold = []
errors_kfold = []
y_preds_kfold = []  # Store predictions for regression plot

np.random.seed(42)

# Bootstrapping and K-Fold Cross-Validation Loop
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fold_preds = []

    # Bootstrapping loop
    for b in range(n_iterations):
        # Sample with replacement from training set
        X_resampled, y_resampled = resample(X_train, y_train, replace=True, random_state=42 + b)

        # Train and predict using SVR
        svr = SVR(kernel='linear')
        svr.fit(X_resampled, y_resampled)
        y_pred = svr.predict(X_test)
        fold_preds.append(y_pred)

    # Average predictions over bootstraps
    y_pred_avg = np.mean(fold_preds, axis=0)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred_avg)
    r2 = r2_score(y_test, y_pred_avg)

    mae_scores_bootstrap.append(mae)
    r2_scores_bootstrap.append(r2)
    errors_bootstrap.extend(y_test - y_pred_avg)

    mae_scores_kfold.append(mae)
    r2_scores_kfold.append(r2)
    errors_kfold.extend(y_test - y_pred_avg)
    y_preds_kfold.extend(y_pred_avg)  # Store the predicted values for regression plot

    print(f"Fold {fold}: MAE = {mae:.4f}, R² = {r2:.4f}")

# Overall performance metrics
print(f"\nOverall Bootstrapping MAE: {np.mean(mae_scores_bootstrap):.4f} ± {np.std(mae_scores_bootstrap):.4f}")
print(f"Overall Bootstrapping R²: {np.mean(r2_scores_bootstrap):.4f} ± {np.std(r2_scores_bootstrap):.4f}")
print(f"Overall K-Fold MAE: {np.mean(mae_scores_kfold):.4f} ± {np.std(mae_scores_kfold):.4f}")
print(f"Overall K-Fold R²: {np.mean(r2_scores_kfold):.4f} ± {np.std(r2_scores_kfold):.4f}")

# --- Plot 1: Error Distribution (Bootstrap) ---
plt.figure(figsize=(8, 5))
plt.hist(errors_bootstrap, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (Bootstrapping)')
plt.xlabel('Error (Actual - Predicted Band Gap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# --- Plot 2: Regression Plot ---
plt.figure(figsize=(8, 6))
sns.regplot(x=y, y=y_preds_kfold, line_kws={"color": "red"}, scatter_kws={"alpha": 0.5})
plt.xlabel('Actual Band Gap (PBE) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('Regression Plot: Predicted vs Actual Band Gap (K-Fold + Bootstrapping)')
plt.grid(True)
plt.show()
