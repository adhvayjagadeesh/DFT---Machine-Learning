import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=[
    'Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'
])

# Separate features and target
X = df.drop(columns=['Band gap (HSE06) [eV]']) 
y = df['Band gap (HSE06) [eV]']

# Convert categorical variables to dummies
X = pd.get_dummies(X, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Initialize model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
rmse_scores = []
r2_scores = []
errors_all = []

for fold, (train_index, test_index) in enumerate(kf.split(X_imputed), 1):
    X_train, X_test = X_imputed[train_index], X_imputed[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    n_samples = X_train.shape[0]
    n_bootstrap = 100
    bootstrap_preds = np.zeros((n_bootstrap, X_test.shape[0]))

    np.random.seed(42 + fold)  # Different seed per fold

    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X_train[bootstrap_idx]
        y_bootstrap = y_train.iloc[bootstrap_idx]

        model.fit(X_bootstrap, y_bootstrap)
        preds = model.predict(X_test)
        bootstrap_preds[i] = preds

    # Aggregate predictions by mean for this fold
    y_pred_fold = bootstrap_preds.mean(axis=0)

    mae_scores.append(mean_absolute_error(y_test, y_pred_fold))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_fold)))
    r2_scores.append(r2_score(y_test, y_pred_fold))
    errors_all.extend(y_test - y_pred_fold)

    print(f"Fold {fold} - MAE: {mae_scores[-1]:.4f}, RMSE: {rmse_scores[-1]:.4f}, R²: {r2_scores[-1]:.4f}")

print(f"\n5-Fold CV with Bootstrapping Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"5-Fold CV with Bootstrapping Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"5-Fold CV with Bootstrapping Mean R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# Plot error distribution
plt.figure(figsize=(8,6))
sns.histplot(errors_all, kde=True, bins=30, color='mediumseagreen', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - 5-Fold CV + Bootstrapping')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Regression plot with combined true and predicted values
y_all_true = np.concatenate([y.iloc[test_idx].values for _, test_idx in kf.split(X_imputed)])
y_all_pred = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(X_imputed), 1):
    X_train, X_test = X_imputed[train_index], X_imputed[test_index]
    y_train = y.iloc[train_index]

    n_samples = X_train.shape[0]
    bootstrap_preds = np.zeros((n_bootstrap, X_test.shape[0]))

    np.random.seed(42 + fold)
    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X_train[bootstrap_idx]
        y_bootstrap = y_train.iloc[bootstrap_idx]

        model.fit(X_bootstrap, y_bootstrap)
        preds = model.predict(X_test)
        bootstrap_preds[i] = preds

    y_pred_fold = bootstrap_preds.mean(axis=0)
    y_all_pred.extend(y_pred_fold)

plt.figure(figsize=(8,6))
sns.regplot(x=y_all_true, y=np.array(y_all_pred), scatter_kws={'s':50}, line_kws={'color':'green'})
plt.title('Regression Plot: True vs Predicted Bandgap - 5-Fold CV + Bootstrapping')
plt.xlabel('True Bandgap')
plt.ylabel('Predicted Bandgap')
plt.grid(True)
plt.show()
