import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=['Band gap (G₀W₀) [eV]', 'Direct band gap (PBE) [eV]', 'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (PBE) [eV]'])
y = df['Band gap (PBE) [eV]']

# Keep only numeric features
X = X.select_dtypes(include=[float, int])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

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

np.random.seed(42)

# Bootstrapping loop
for i in range(n_iterations):
    # Resample with replacement for bootstrapping
    X_resampled, y_resampled = resample(X_scaled, y, replace=True, random_state=42 + i)

    # Split into train/test sets
    split_index = int(0.8 * len(X_resampled))
    X_train, X_test = X_resampled[:split_index], X_resampled[split_index:]
    y_train, y_test = y_resampled[:split_index], y_resampled[split_index:]

    # Train SVR model
