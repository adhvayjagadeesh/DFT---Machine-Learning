import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=[
  'Direct band gap (PBE) [eV]', 'Direct band gap (PBE) [eV].1',
  'Band gap (PBE) [eV]', 'Band gap (G₀W₀) [eV]',
  'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]',
  'Direct band gap (HSE06) [eV].1', 'CBM wrt. vacuum (PBE) [eV]',
  'VBM wrt. vacuum (PBE) [eV]'
])

target = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='mean')),
  ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
  ('num', numeric_transformer, numeric_cols),
  ('cat', categorical_transformer, categorical_cols)
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
mae_scores = []
rmse_scores = []

all_y_test = []
all_y_pred = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
  print(f"\nFold {fold_idx+1}")

  X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
  y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

  model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))
  ])

  model_pipeline.fit(X_train, y_train)
  y_pred = model_pipeline.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))

  print(f"  R²  : {r2:.4f}")
  print(f"  MAE : {mae:.4f}")
  print(f"  RMSE: {rmse:.4f}")

  r2_scores.append(r2)
  mae_scores.append(mae)
  rmse_scores.append(rmse)

  all_y_test.extend(y_test)
  all_y_pred.extend(y_pred)

print("\nAverage CV scores:")
print(f"Mean R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Mean MAE : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

# Plot Actual vs Predicted aggregated over folds
plt.figure(figsize=(8, 6))
sns.regplot(x=all_y_test, y=all_y_pred, line_kws={"color": "red"}, scatter_kws={"alpha": 0.4, "edgecolor":"k"})
plt.xlabel('Actual Band Gap (HSE06) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('SVR 5-Fold CV: Actual vs Predicted Band Gap')
plt.grid(True)
plt.tight_layout()
plt.show()

# Error distribution histogram
errors = np.array(all_y_test) - np.array(all_y_pred)
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=40, edgecolor='k', alpha=0.7)
plt.title('SVR 5-Fold CV: Prediction Error Distribution')
plt.xlabel('Error (Actual - Predicted Band Gap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
