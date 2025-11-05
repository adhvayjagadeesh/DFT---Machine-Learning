import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold

# Load the dataset
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

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col].astype(str))
  label_encoders[col] = le

# Fill missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features and target
X = df.drop(columns=["Band gap (HSE06) [eV]"])
y = df["Band gap (HSE06) [eV]"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_list = []
r2_list = []
rmse_list = []

fold = 1
for train_index, test_index in kf.split(X_scaled):
  X_train, X_test = X_scaled[train_index], X_scaled[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))

  mae_list.append(mae)
  r2_list.append(r2)
  rmse_list.append(rmse)

  print(f"Fold {fold}: MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
  fold += 1

print(f"\nAverage MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Average RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
print(f"Average R² : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# Plot results from last fold
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Band Gap (HSE06) [eV]")
plt.ylabel("Predicted Band Gap (HSE06) [eV]")
plt.title("Actual vs Predicted Band Gap (Last Fold)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.grid(True)
plt.show()
