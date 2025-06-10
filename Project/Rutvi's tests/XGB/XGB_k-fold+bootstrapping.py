import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'])


# Encode categorical features if any
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Fill remaining missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features and target
X = df.drop(columns=["Band gap (HSE06) [eV]"])
y = df["Band gap (HSE06) [eV]"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstrap = 50

mae_scores = []
r2_scores = []

# K-Fold + Bootstrapping
fold_idx = 1
for train_idx, test_idx in kf.split(X_scaled):
    print(f"\nFold {fold_idx}...")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[bootstrap_idx]
        y_boot = y_train.iloc[bootstrap_idx]

        model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=i)
        model.fit(X_boot, y_boot)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        r2_scores.append(r2)

        # Save predictions from last model for plotting
        if fold_idx == 5 and i == n_bootstrap - 1:
            final_y_test = y_test
            final_y_pred = y_pred
    fold_idx += 1

# Results summary
print(f"\nBootstrapped K-Fold MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Bootstrapped K-Fold R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# Plot regression graph (last fold's last bootstrap model)
plt.scatter(final_y_test, final_y_pred, alpha=0.5)
plt.xlabel("Actual Band Gap (HSE06) [eV]")
plt.ylabel("Predicted Band Gap (HSE06) [eV]")
plt.title("Actual vs Predicted Band Gap (HSE06) [Final Fold - Last Bootstrap]")
plt.plot([min(final_y_test), max(final_y_test)], [min(final_y_test), max(final_y_test)], color='red', linestyle='dashed')
plt.grid(True)
plt.show()

# --- Plot Error Distribution (Bar Chart / Histogram) ---
errors = final_y_test - final_y_pred

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.title("Prediction Error Distribution (Final Fold - Last Bootstrap)")
plt.xlabel("Error (Actual - Predicted Band Gap)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

