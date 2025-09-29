# type: ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from scipy.optimize import minimize
from scipy.stats import spearmanr
from data.c2db import df

# Drop columns with >90% missing data + identifiers
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Fill missing numerical values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
X = df.drop(columns=["Band gap (HSE06) [eV]"])
y = df["Band gap (HSE06) [eV]"]

# Train-test split
X_train, X_holdout, y_train_full, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for predictions
all_true = []
xgb_all = []
rf_all = []
mlp_all = []
blr_all = []

# Cross-validation loop
for train_idx, val_idx in kf.split(X_train_scaled):
    X_train, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    # Define models
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    blr = BayesianRidge()

    # Train models
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    blr.fit(X_train, y_train)

    # Predict
    xgb_pred = xgb.predict(X_val)
    rf_pred = rf.predict(X_val)
    mlp_pred = mlp.predict(X_val)
    blr_pred = blr.predict(X_val)

    # Store results
    all_true.extend(y_val)
    xgb_all.extend(xgb_pred)
    rf_all.extend(rf_pred)
    mlp_all.extend(mlp_pred)
    blr_all.extend(blr_pred)

# Convert predictions to numpy arrays
y_true = np.array(all_true)
pred_matrix = np.vstack([xgb_all, rf_all, mlp_all, blr_all]).T

# Define loss and optimize weights
def loss_fn(weights):
    blended = np.dot(pred_matrix, weights)
    return mean_absolute_error(y_true, blended)

init_weights = [1/4] * 4
bounds = [(0, 1)] * 4
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

result = minimize(loss_fn, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Final training on full data
xgb_final = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
mlp_final = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
blr_final = BayesianRidge()

xgb_final.fit(X_train_scaled, y_train_full)
rf_final.fit(X_train_scaled, y_train_full)
mlp_final.fit(X_train_scaled, y_train_full)
blr_final.fit(X_train_scaled, y_train_full)

# Final predictions on holdout set
xgb_pred_final = xgb_final.predict(X_holdout_scaled)
rf_pred_final = rf_final.predict(X_holdout_scaled)
mlp_pred_final = mlp_final.predict(X_holdout_scaled)
blr_pred_final = blr_final.predict(X_holdout_scaled)

# Hybrid prediction
pred_matrix_final = np.vstack([xgb_pred_final, rf_pred_final, mlp_pred_final, blr_pred_final]).T

hybrid_pred = np.dot(pred_matrix_final, optimal_weights)

# Exports for visualization
y_true = y_holdout
y_pred = hybrid_pred
k = X.shape[1]
