import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=[
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (PBE) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]'
])

# Target column
target_col = 'Band gap (PBE) [eV]'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
feature_names = imputer.get_feature_names_out(X.columns)
X = pd.DataFrame(X_imputed, columns=feature_names)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit LASSO with cross-validation to choose best alpha
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha selected by LASSO CV: {lasso_cv.alpha_:.6f}")

# -------------------------------
# K-Fold Cross-Validation
# -------------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

mae_scores = []
r2_scores = []

# Plot setup
fig, axs = plt.subplots(1, k, figsize=(18, 4), sharey=True)
fig.suptitle('Actual vs Predicted Bandgap per Fold', fontsize=16)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_kf_train, X_kf_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_kf_train, y_kf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = Lasso(alpha=lasso_cv.alpha_)
    model.fit(X_kf_train, y_kf_train)

    y_kf_pred = model.predict(X_kf_val)
    fold_mae = mean_absolute_error(y_kf_val, y_kf_pred)
    fold_r2 = r2_score(y_kf_val, y_kf_pred)

    mae_scores.append(fold_mae)
    r2_scores.append(fold_r2)

    # Plot each fold
    ax = axs[fold]
    ax.scatter(y_kf_val, y_kf_pred, alpha=0.6, edgecolors='k')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_title(f'Fold {fold+1}\nMAE={fold_mae:.3f}, R²={fold_r2:.3f}')
    ax.set_xlabel('Actual')
    if fold == 0:
        ax.set_ylabel('Predicted')
    ax.grid(True)

    # Plot error distribution for this fold
    fold_errors = y_kf_val - y_kf_pred
    plt.figure(figsize=(5, 3))
    plt.hist(fold_errors, bins=20, edgecolor='k', alpha=0.7)
    plt.title(f'Fold {fold+1} Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print(f"\nAverage MAE across {k} folds: {sum(mae_scores)/k:.4f}")
print(f"Average R² across {k} folds: {sum(r2_scores)/k:.4f}")

# -------------------------------
# Final Evaluation on Test Set
# -------------------------------
final_model = Lasso(alpha=lasso_cv.alpha_)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nFinal Test Set Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot: Actual vs Predicted on Test Set
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Bandgap [eV]')
plt.ylabel('Predicted Bandgap [eV]')
plt.title('LASSO: Actual vs Predicted Bandgap (Test Set)')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Feature Importance
# -------------------------------
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': final_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 most influential features:")
print(coef_df.head(10))

# -------------------------------
# Plot error distribution on test set
# -------------------------------
test_errors = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.hist(test_errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (Test Set)')
plt.xlabel('Error (Actual - Predicted Bandgap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
