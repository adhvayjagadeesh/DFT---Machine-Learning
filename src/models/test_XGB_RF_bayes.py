"""
XGB_RF_bayes.py - nested CV + OOF stacking with y_test collection, metrics, and plots.

Outputs:
 - prints per-fold metrics for XGB, RF, and META
 - aggregated metrics (RMSE, MAE, R2)
 - saves final models to models/
 - saves figures to models/pred_vs_true.png
 - saves CSV of predictions to models/predictions.csv

Notes: robust to sklearn versions that lack 'squared' kw in mean_squared_error;
uses RMSE = sqrt(MSE). Uses safe single-threaded OOF to avoid multiprocessing issues.
"""

import numpy as np
import os, time, sys
import multiprocessing
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# import your dataset variables from final.py
from data.final import x_, y_, k_

# --------------------------
# Config
# --------------------------
RNG_SEED = 67
np.random.seed(RNG_SEED)

# CPU detection
CPU_COUNT = max(1, multiprocessing.cpu_count())
# keep OOF safe (set to >1 if you know your env supports it reliably)
PARALLEL_PRED_JOBS = 1
BAYES_N_JOBS = 1

# ------------------------
# Pipelines & param spaces
# ------------------------
pipe_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(
        objective="reg:squarederror",
        random_state=RNG_SEED,
        tree_method="hist",
        n_jobs=1,
        use_label_encoder=False,
        verbosity=0,
        eval_metric="rmse"
    ))
])

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(random_state=RNG_SEED, n_jobs=1))
])

hyperparams_xgb = {
    "xgb__max_depth": Integer(3, 8),
    "xgb__min_child_weight": Integer(1, 8),
    "xgb__learning_rate": Real(1e-2, 1e-1, prior="log-uniform"),
    "xgb__n_estimators": Integer(100, 500),
    "xgb__subsample": Real(0.7, 1.0),
    "xgb__colsample_bytree": Real(0.5, 1.0),
    "xgb__reg_alpha": Real(1e-6, 0.5, prior="log-uniform"),
    "xgb__reg_lambda": Real(0.5, 2.0, prior="log-uniform"),
    "xgb__gamma": Real(0.0, 2.0),
}

hyperparams_rf = {
    "rf__n_estimators": Integer(100, 500),
    "rf__max_depth": Categorical([None, 10, 25, 50]),
    "rf__min_samples_split": Integer(2, 12),
    "rf__min_samples_leaf": Integer(1, 6),
    "rf__max_features": Categorical(["sqrt", "log2"]),
    "rf__bootstrap": Categorical([True, False]),
}

# ------------------------
# CV params
# ------------------------
outer_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
inner_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
n_iter_bayes = 20
verbose_bayes = 1

def make_bayes_xgb(n_iter=n_iter_bayes):
    return BayesSearchCV(
        pipe_xgb, hyperparams_xgb,
        cv=inner_k, n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=verbose_bayes, random_state=RNG_SEED
    )

def make_bayes_rf(n_iter=n_iter_bayes):
    return BayesSearchCV(
        pipe_rf, hyperparams_rf,
        cv=inner_k, n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=verbose_bayes, random_state=RNG_SEED
    )

# ------------------------------
# Prepare data & containers
# ------------------------------
X = x_.copy().reset_index(drop=True)
y = y_.copy().reset_index(drop=True)

outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=RNG_SEED)

# arrays to mimic your original script's behavior
y_pred = np.array([])   # concatenated blended predictions across outer folds
y_test = np.array([])   # concatenated ground-truth across outer folds

fold_results = []
start_time = time.time()

print(f"Starting nested CV: outer_k={outer_k}, inner_k={inner_k}, n_iter_bayes={n_iter_bayes}")

for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
    t0 = time.time()
    print("\n" + "="*60)
    print(f"OUTER FOLD {fold_idx}/{outer_k}")
    print("="*60)

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- Inner tuning ---
    print("Tuning XGBoost (inner CV)...")
    bayes_xgb = make_bayes_xgb()
    bayes_xgb.fit(X_train, y_train)
    best_xgb = bayes_xgb.best_estimator_
    print(" -> Best XGB params:", bayes_xgb.best_params_)

    print("Tuning RandomForest (inner CV)...")
    bayes_rf = make_bayes_rf()
    bayes_rf.fit(X_train, y_train)
    best_rf = bayes_rf.best_estimator_
    print(" -> Best RF params:", bayes_rf.best_params_)

    # --- OOF preds for meta training (safe single-threaded)
    print("Generating OOF predictions for meta training (single-threaded safe mode)...")
    oof_xgb = cross_val_predict(best_xgb, X_train, y_train, cv=inner_k, method="predict", n_jobs=1)
    oof_rf  = cross_val_predict(best_rf,  X_train, y_train, cv=inner_k, method="predict", n_jobs=1)

    meta_X_train = np.vstack([oof_xgb, oof_rf]).T
    meta = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta.fit(meta_X_train, y_train)

    # --- Predictions on outer validation split ---
    preds_xgb = best_xgb.predict(X_val)
    preds_rf  = best_rf.predict(X_val)
    preds_blend = meta.predict(np.vstack([preds_xgb, preds_rf]).T)  # meta learner blend (preferred)
    # if you prefer simple average: preds_blend = (preds_xgb + preds_rf) / 2.0

    # mimic your original accumulators
    y_test = np.concatenate([y_test, y_val.values])
    y_pred = np.concatenate([y_pred, preds_blend])

    # compute robust metrics (RMSE = sqrt(MSE))
    def metrics(y_t, y_p):
        y_t = np.asarray(y_t)
        y_p = np.asarray(y_p)
        mse = mean_squared_error(y_t, y_p)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    m_xgb = metrics(y_val, preds_xgb)
    m_rf  = metrics(y_val, preds_rf)
    m_meta = metrics(y_val, preds_blend)

    print(f"Fold {fold_idx} — XGB: RMSE={m_xgb['rmse']:.6f}, MAE={m_xgb['mae']:.6f}, R2={m_xgb['r2']:.6f}")
    print(f"Fold {fold_idx} — RF : RMSE={m_rf['rmse']:.6f}, MAE={m_rf['mae']:.6f}, R2={m_rf['r2']:.6f}")
    print(f"Fold {fold_idx} — META: RMSE={m_meta['rmse']:.6f}, MAE={m_meta['mae']:.6f}, R2={m_meta['r2']:.6f}")

    # meta coefficients
    print("Meta coefficients:", getattr(meta, "coef_", None), "intercept:", getattr(meta, "intercept_", None))

    fold_results.append({
        "fold": fold_idx,
        "xgb": m_xgb,
        "rf": m_rf,
        "meta": m_meta,
        "meta_coef": getattr(meta, "coef_", None).tolist() if getattr(meta, "coef_", None) is not None else None
    })

    print(f"Time for fold {fold_idx}: {time.time() - t0:.1f}s")

# -------------------------
# Aggregated metrics (like original script might report)
# -------------------------
agg_mse = mean_squared_error(y_test, y_pred)
agg_rmse = float(np.sqrt(agg_mse))
agg_mae  = float(mean_absolute_error(y_test, y_pred))
agg_r2   = float(r2_score(y_test, y_pred))

print("\n" + "#"*60)
print("Aggregated results across all outer folds:")
print(f"Aggregated RMSE: {agg_rmse:.6f}")
print(f"Aggregated MAE : {agg_mae:.6f}")
print(f"Aggregated R2  : {agg_r2:.6f}")
print("#"*60)

# -------------------------
# Save predictions CSV
# -------------------------
os.makedirs("models", exist_ok=True)
pred_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
pred_df["residual"] = pred_df["y_test"] - pred_df["y_pred"]
pred_csv_path = os.path.join("models", "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"Saved predictions CSV -> {pred_csv_path}")

# -------------------------
# Plot: predicted vs true + residuals
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# scatter predicted vs true
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.6, s=20)
minv = min(np.min(y_test), np.min(y_pred))
maxv = max(np.max(y_test), np.max(y_pred))
ax.plot([minv, maxv], [minv, maxv], color="k", linestyle="--", linewidth=1)
ax.set_xlabel("True (y_test)")
ax.set_ylabel("Predicted (y_pred)")
ax.set_title(f"Pred vs True (RMSE={agg_rmse:.4f})")

# residual histogram
ax2 = axes[1]
residuals = y_test - y_pred
ax2.hist(residuals, bins=40, alpha=0.8)
ax2.set_title("Residuals histogram")
ax2.set_xlabel("Residual (true - pred)")
ax2.set_ylabel("Count")

plt.tight_layout()
fig_path = os.path.join("models", "pred_vs_true.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved figure -> {fig_path}")
try:
    plt.show()
except Exception:
    # some headless environments can't show -> continue
    pass

# -------------------------
# Optional: final refit on full dataset and save final models
# -------------------------
REFIT_FINAL = True
FINAL_N_ITER = max(n_iter_bayes, 30)

if REFIT_FINAL:
    print("\nRefitting tuned models on full dataset (this can take time)...")
    t_ref = time.time()
    bayes_xgb_full = make_bayes_xgb(n_iter=FINAL_N_ITER)
    bayes_rf_full = make_bayes_rf(n_iter=FINAL_N_ITER)
    bayes_xgb_full.fit(X, y)
    bayes_rf_full.fit(X, y)
    best_xgb_full = bayes_xgb_full.best_estimator_
    best_rf_full  = bayes_rf_full.best_estimator_
    print("Final best XGB params:", bayes_xgb_full.best_params_)
    print("Final best RF params :", bayes_rf_full.best_params_)

    oof_xgb_full = cross_val_predict(best_xgb_full, X, y, cv=inner_k, method="predict", n_jobs=1)
    oof_rf_full  = cross_val_predict(best_rf_full,  X, y, cv=inner_k, method="predict", n_jobs=1)
    meta_X_full = np.vstack([oof_xgb_full, oof_rf_full]).T
    meta_final = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta_final.fit(meta_X_full, y)

    joblib.dump(best_xgb_full, os.path.join("models", "best_xgb_full.joblib"))
    joblib.dump(best_rf_full,  os.path.join("models", "best_rf_full.joblib"))
    joblib.dump(meta_final,    os.path.join("models", "meta_final.joblib"))
    print(f"Saved final models to models/ (time: {time.time()-t_ref:.1f}s)")

print("Script finished.")