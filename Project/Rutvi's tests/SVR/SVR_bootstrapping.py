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
from sklearn.utils import resample

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

# Initialize lists to collect metrics for each bootstrap
r2_scores = []
mae_scores = []
rmse_scores = []

n_bootstraps = 100
np.random.seed(42)

for i in range(n_bootstraps):
    # Bootstrap sampling with replacement
    X_resampled, y_resampled = resample(X, y, replace=True, random_state=42 + i)
    # Use out-of-bag samples as test set (those NOT in bootstrap sample)
    mask = ~X.index.isin(X_resampled.index)
    X_test = X.loc[mask]
    y_test = y.loc[mask]
    
    # If no test samples (rare but possible), skip this bootstrap
    if len(X_test) == 0:
        continue

    # Define and fit the pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel='rbf'))
    ])
    model_pipeline.fit(X_resampled, y_resampled)

    y_pred = model_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{n_bootstraps} bootstraps")

print("\nBootstrap CV results (100 bootstraps):")
print(f"Mean R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Mean MAE : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")


