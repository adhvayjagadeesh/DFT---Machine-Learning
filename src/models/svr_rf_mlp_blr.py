# type: ignore

### Reduced overfitting with BLR added ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from scipy.optimize import minimize
from data.final import k_fold

# Storage for true and predicted values
all_true = []   
svr_all = []
rf_all = []
mlp_all = []
blr_all = []

# Cross-validation loop to find optimal weights
for x_train, y_train, x_test, y_test in k_fold():
    # Define models
    svr = SVR(kernel='rbf', C=10, epsilon=0.1)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    blr = BayesianRidge()

    # Train models
    svr.fit(x_train, y_train)
    rf.fit(x_train, y_train)
    mlp.fit(x_train, y_train)
    blr.fit(x_train, y_train)

    # Predict
    svr_pred = svr.predict(x_test)
    rf_pred = rf.predict(x_test)
    mlp_pred = mlp.predict(x_test)
    blr_pred = blr.predict(x_test)

    # Store predictions
    all_true.extend(y_test)
    svr_all.extend(svr_pred)
    rf_all.extend(rf_pred)
    mlp_all.extend(mlp_pred)
    blr_all.extend(blr_pred)

# Convert to arrays
y_test = np.array(all_true)
pred_matrix = np.vstack([svr_all, rf_all, mlp_all, blr_all]).T

# Optimize weights for hybrid model
def loss_fn(weights):
    blended = np.dot(pred_matrix, weights)
    return mean_absolute_error(y_test, blended)

init_weights = [1/4, 1/4, 1/4, 1/4]
bounds = [(0, 1)] * 4
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

result = minimize(loss_fn, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Retrain models on full training data and evaluate on holdout
svr_final = SVR(kernel='rbf', C=10, epsilon=0.1)
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
mlp_final = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
blr_final = BayesianRidge()

svr_final.fit(x_train_scaled, y_train_full)
rf_final.fit(x_train_scaled, y_train_full)
mlp_final.fit(x_train_scaled, y_train_full)
blr_final.fit(x_train_scaled, y_train_full)

# Final predictions on holdout set
svr_pred_final = svr_final.predict(X_holdout_scaled)
rf_pred_final = rf_final.predict(X_holdout_scaled)
mlp_pred_final = mlp_final.predict(X_holdout_scaled)
blr_pred_final = blr_final.predict(X_holdout_scaled)

# Hybrid prediction
pred_matrix_final = np.vstack([svr_pred_final, rf_pred_final, mlp_pred_final, blr_pred_final]).T
y_pred = np.dot(pred_matrix_final, optimal_weights)