# type: ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.optimize import minimize
from scipy.stats import spearmanr
import xgboost as xgb
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

# Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for true and predicted values
all_true = []
svr_all = []
rf_all = []
xgb_all = []
blr_all = []

# Cross-validation loop
for train_idx, val_idx in kf.split(X_train_scaled):
    X_train, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    # Define models
    svr = SVR(kernel='rbf', C=10, epsilon=0.1)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    blr = BayesianRidge()

    # Train models
    svr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    blr.fit(X_train, y_train)

    # Predict
    svr_pred = svr.predict(X_val)
    rf_pred = rf.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    blr_pred = blr.predict(X_val)

    # Store predictions
    all_true.extend(y_val)
    svr_all.extend(svr_pred)
    rf_all.extend(rf_pred)
    xgb_all.extend(xgb_pred)
    blr_all.extend(blr_pred)

# Convert to arrays
y_true = np.array(all_true)
pred_matrix = np.vstack([svr_all, rf_all, xgb_all, blr_all]).T

# Optimize weights for hybrid model
def loss_fn(weights):
    blended = np.dot(pred_matrix, weights)
    return mean_absolute_error(y_true, blended)

init_weights = [1/4, 1/4, 1/4, 1/4]
bounds = [(0, 1)] * 4
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

result = minimize(loss_fn, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Retrain models on full training data and evaluate on holdout
svr_final = SVR(kernel='rbf', C=10, epsilon=0.1)
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
blr_final = BayesianRidge()

svr_final.fit(X_train_scaled, y_train_full)
rf_final.fit(X_train_scaled, y_train_full)
xgb_final.fit(X_train_scaled, y_train_full)
blr_final.fit(X_train_scaled, y_train_full)

# Final predictions on holdout set
svr_pred_final = svr_final.predict(X_holdout_scaled)
rf_pred_final = rf_final.predict(X_holdout_scaled)
xgb_pred_final = xgb_final.predict(X_holdout_scaled)
blr_pred_final = blr_final.predict(X_holdout_scaled)

# Hybrid prediction
pred_matrix_final = np.vstack([svr_pred_final, rf_pred_final, xgb_pred_final, blr_pred_final]).T

hybrid_pred = np.dot(pred_matrix_final, optimal_weights)

# Exports for visualization
y_true = y_holdout
y_pred = hybrid_pred
k = X.shape[1]