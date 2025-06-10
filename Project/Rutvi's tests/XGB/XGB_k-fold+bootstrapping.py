import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold

# Load dataset and preprocess (same as above)
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'])

drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=["Formula"], inplace=True)
df.drop(columns=drop_cols, inplace=True, errors='ignore')

cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop(columns=["Band gap (HSE06) [eV]"])
y = df["Band gap (HSE06) [eV]"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstraps = 100
np.random.seed(42)

fold_mae_means = []
fold_mae_stds = []
fold_rmse_means = []
fold_rmse_stds = []
fold_r2_means = []
fold_r2_stds = []

fold_num = 1
for train_index, test_index in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    bootstrap_mae = []
    bootstrap_rmse = []
    bootstrap_r2 = []

    for b in range(n_bootstraps):
        # Bootstrap resample within train fold
        indices = np.random.choice(len(X_train_fold), size=len(X_train_fold), replace=True)
        X_boot = X_train_fold[indices]
        y_boot = y_train_fold.iloc[indices]

        model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_boot, y_boot)

        y_pred = model.predict(X_test_fold)
        mae = mean_absolute_error(y_test_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
        r2 = r2_score(y_test_fold, y_pred)

        bootstrap_mae.append(mae)
        bootstrap_rmse.append(rmse)
        bootstrap_r2.append(r2)

    fold_mae_means.append(np.mean(bootstrap_mae))
    fold_mae_stds.append(np.std(bootstrap_mae))
    fold_rmse_means.append(np.mean(bootstrap_rmse))
    fold_rmse_stds.append(np.std(bootstrap_rmse))
    fold_r2_means.append(np.mean(bootstrap_r2))
    fold_r2_stds.append(np.std(bootstrap_r2))

    print(f"Fold {fold_num}: MAE = {np.mean(bootstrap_mae):.4f} ± {np.std(bootstrap_mae):.4f}, "
          f"RMSE = {np.mean(bootstrap_rmse):.4f} ± {np.std(bootstrap_rmse):.4f}, "
          f"R² = {np.mean(bootstrap_r2):.4f} ± {np.std(bootstrap_r2):.4f}")
    fold_num += 1

print("\nOverall CV Results:")
print(f"MAE: {np.mean(fold_mae_means):.4f} ± {np.sqrt(np.sum(np.array(fold_mae_stds)**2)/len(fold_mae_stds)):.4f}")
print(f"RMSE: {np.mean(fold_rmse_means):.4f} ± {np.sqrt(np.sum(np.array(fold_rmse_stds)**2)/len(fold_rmse_stds)):.4f}")
print(f"R² : {np.mean(fold_r2_means):.4f} ± {np.sqrt(np.sum(np.array(fold_r2_stds)**2)/len(fold_r2_stds)):.4f}")

# Plot results from last fold's last bootstrap
plt.scatter(y_test_fold, y_pred, alpha=0.5)
plt.xlabel("Actual Band Gap (HSE06) [eV]")
plt.ylabel("Predicted Band Gap (HSE06) [eV]")
plt.title("Actual vs Predicted Band Gap (Last Fold, Last Bootstrap)")
plt.plot([min(y_test_fold), max(y_test_fold)], [min(y_test_fold), max(y_test_fold)], color='red', linestyle='dashed')
plt.grid(True)
plt.show()
