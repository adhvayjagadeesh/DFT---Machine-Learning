import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load and prepare data
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]',])

# Define target
target = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target])
y = df[target]

# Encode and clean features
X = pd.get_dummies(X)                         # One-hot encoding
X = X.select_dtypes(include=[np.number])      # Keep only numeric
X = X.dropna(axis=1, how='all')               # Drop columns with all NaNs
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)  # Impute remaining NaNs

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Get feature names (for later interpretation)
feature_names = X.columns

# LASSO with cross-validation to select alpha
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)
optimal_alpha = lasso_cv.alpha_
print(f"\nOptimal alpha from LASSO CV: {optimal_alpha:.6f}")

# ------------------------------------------
# K-Fold + Bootstrapping
# ------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstraps = 100

all_y_true = []
all_y_pred = []
mae_list = []
r2_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    fold_preds = []

    for b in range(n_bootstraps):
        boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[boot_idx]
        y_boot = y_train.iloc[boot_idx]

        model = Lasso(alpha=optimal_alpha)
        model.fit(X_boot, y_boot)
        y_pred = model.predict(X_test)
        fold_preds.append(y_pred)

    y_pred_avg = np.mean(fold_preds, axis=0)

    mae = mean_absolute_error(y_test, y_pred_avg)
    r2 = r2_score(y_test, y_pred_avg)

    mae_list.append(mae)
    r2_list.append(r2)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred_avg)

    print(f"Fold {fold}: MAE = {mae:.4f}, R² = {r2:.4f}")

# ------------------------------------------
# Overall performance
# ------------------------------------------
print(f"\nOverall MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Overall R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE : {rmse:.4f}")

# Error analysis
errors = np.array(all_y_true) - np.array(all_y_pred)

# Error bar chart
plt.figure(figsize=(8, 5))
plt.bar(range(len(errors)), errors, color='steelblue', edgecolor='k', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Prediction Errors (Actual - Predicted Band Gap)')
plt.xlabel('Sample Index')
plt.ylabel('Error [eV]')
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()

# Regression plot
plt.figure(figsize=(8, 6))
sns.regplot(x=all_y_true, y=all_y_pred, line_kws={"color": "red"}, scatter_kws={"alpha": 0.5})
plt.xlabel('Actual Band Gap (HSE06) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('Regression Plot: Predicted vs Actual Band Gap')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------
# Feature importance from final model
# ------------------------------------------
final_model = Lasso(alpha=optimal_alpha)
final_model.fit(X_scaled, y)
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': final_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Influential Features:")
print(coef_df.head(10))
