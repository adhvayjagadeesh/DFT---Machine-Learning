"""
hybrid_gbt_svr_bayes_improved.py

Improvements applied (see your instructions):
 - Nested CV (outer KFold = k_, inner BayesSearchCV)
 - Out-of-fold (OOF) stacking with Ridge meta-learner (learned blend)
 - Safe single-threaded OOF & safe Bayes defaults to avoid nested-parallelism issues
 - Robust sanitization of features/target (replace inf, fill NaN)
 - Per-fold + aggregated metrics, predictions CSV, diagnostic plot saved to models/
 - Optional final refit on full data and saving of final tuned models
 - Keeps GBT and SVR hyperparameter spaces exactly as provided in the original script
"""

import os
import time
import sys
import numpy as np
import pandas as pd
import multiprocessing
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# import dataset variables from final.py
from data.final import x_, y_, k_

# -------------------------
# Helpers: sanitize X/y
# -------------------------
def sanitize_features_and_target(X, y):
    """
    Replace +/-inf with NaN, fill NaN in features with column mean (or 0 if entirely NaN).
    Ensure y contains no NaN/Inf (raise useful error).
    Accepts pandas DataFrame/Series or numpy arrays.
    Returns numpy arrays (X_clean, y_clean).
    """
    # Features
    if isinstance(X, pd.DataFrame):
        Xc = X.copy()
        Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in Xc.columns:
            col_vals = Xc[col]
            if col_vals.isnull().all():
                Xc[col] = 0.0
            else:
                Xc[col].fillna(col_vals.mean(), inplace=True)
        X_out = Xc.values.astype(float)
    else:
        Xa = np.asarray(X, dtype=float).copy()
        Xa[~np.isfinite(Xa)] = np.nan
        col_means = np.nanmean(Xa, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(Xa))
        if inds[0].size > 0:
            Xa[inds] = np.take(col_means, inds[1])
        X_out = Xa

    # Target
    y_out = np.asarray(y, dtype=float)
    if not np.isfinite(y_out).all():
        if np.isnan(y_out).any():
            raise ValueError("Target y contains NaN values. Clean target before training.")
        else:
            raise ValueError("Target y contains non-finite values (inf). Clean target before training.")
    return X_out, y_out

# -------------------------
# Configuration / safety
# -------------------------
RNG_SEED = 67
np.random.seed(RNG_SEED)

CPU_COUNT = max(1, multiprocessing.cpu_count())
# Safe defaults to avoid nested-parallelism issues. Increase only if you know your env is safe.
PARALLEL_PRED_JOBS = 1
BAYES_N_JOBS = 1

# Toggle DEBUG for quick smoke tests (set True for fast run)
DEBUG = False

# CV and Bayes settings
if DEBUG:
    outer_k = 3
    inner_k = 2
    n_iter_bayes = 4
    REFIT_FINAL = False
else:
    outer_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
    inner_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
    n_iter_bayes = 20
    REFIT_FINAL = True

verbose_bayes = 2

# -------------------------
# Pipelines (scaling inside pipelines)
# -------------------------
pipe_gbt = Pipeline([
    ("scaler", StandardScaler()),
    ("gbt", GradientBoostingRegressor(random_state=RNG_SEED))
])

pipe_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(max_iter=1000000))  # keep your SVR instantiation
])

# -------------------------
# Hyperparameter spaces (kept exactly as provided)
# -------------------------
hyperparams_gbt = {
    "gbt__n_estimators": Integer(200, 800),
    "gbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
    "gbt__max_depth": Integer(3, 10),
    "gbt__min_samples_split": Integer(2, 15),
    "gbt__min_samples_leaf": Integer(1, 10),
    "gbt__subsample": Real(0.7, 1),
    "gbt__max_features": Categorical(["sqrt", 0.7, None]),
}

hyperparams_svr = {
    "svr__C": Real(1e-3, 1e+6, prior="log-uniform"),
    "svr__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
    "svr__degree": Integer(1, 9),
    "svr__epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
    "svr__kernel": Categorical(["linear", "poly", "rbf"]),
}

# -------------------------
# BayesSearchCV factories (safe n_jobs)
# -------------------------
def make_bayes_gbt(n_iter=n_iter_bayes):
    return BayesSearchCV(
        pipe_gbt,
        hyperparams_gbt,
        cv=inner_k,
        n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=verbose_bayes,
        random_state=RNG_SEED,
        error_score=np.nan
    )

def make_bayes_svr(n_iter=n_iter_bayes):
    return BayesSearchCV(
        pipe_svr,
        hyperparams_svr,
        cv=inner_k,
        n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=verbose_bayes,
        random_state=RNG_SEED,
        error_score=np.nan
    )

# -------------------------
# Prepare data & containers
# -------------------------
X = x_.copy().reset_index(drop=True)
y = y_.copy().reset_index(drop=True)

outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=RNG_SEED)

# Keep original-style accumulators (same semantics as original script)
y_pred = np.array([])   # concatenated hybrid predictions across outer folds
y_test = np.array([])   # concatenated ground-truth across outer folds

fold_results = []
start_time = time.time()

print(f"Starting nested CV: outer_k={outer_k}, inner_k={inner_k}, n_iter_bayes={n_iter_bayes}")

# -------------------------
# Outer CV loop: nested workflow
# -------------------------
for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
    t0 = time.time()
    print("\n" + "="*60)
    print(f"OUTER FOLD {fold_idx}/{outer_k}")
    print("="*60)

    X_train_df, X_val_df = X.iloc[train_idx], X.iloc[val_idx]
    y_train_ser, y_val_ser = y.iloc[train_idx], y.iloc[val_idx]

    # sanitize training partition
    X_train_arr, y_train_arr = sanitize_features_and_target(X_train_df, y_train_ser)

    # -------------------------
    # Inner tuning: GBT
    # -------------------------
    print("Tuning GradientBoosting (inner CV)...")
    bayes_gbt = make_bayes_gbt()
    bayes_gbt.fit(X_train_arr, y_train_arr)
    best_gbt = bayes_gbt.best_estimator_
    print(" -> Best GBT params:", getattr(bayes_gbt, "best_params_", None))

    # -------------------------
    # Inner tuning: SVR
    # -------------------------
    print("Tuning SVR (inner CV)...")
    bayes_svr = make_bayes_svr()
    try:
        bayes_svr.fit(X_train_arr, y_train_arr)
        best_svr = bayes_svr.best_estimator_
        print(" -> Best SVR params:", getattr(bayes_svr, "best_params_", None))
    except Exception as e:
        # fallback: log and use a safe default SVR pipeline for this fold
        print("Warning: SVR BayesSearchCV failed for this fold with exception:", repr(e))
        print("Falling back to a safe default SVR (C=1.0, kernel='rbf', gamma='scale').")
        safe_svr = SVR(C=1.0, kernel="rbf", gamma="scale", epsilon=0.1, max_iter=1000000)
        best_svr = Pipeline([("scaler", StandardScaler()), ("svr", safe_svr)])
        best_svr.fit(X_train_arr, y_train_arr)

    # -------------------------
    # Create OOF predictions on the training partition for meta training
    # -------------------------
    print("Generating out-of-fold (OOF) predictions for meta training (single-threaded safe)...")
    # cross_val_predict clones the tuned estimators; pass arrays (X_train_arr)
    oof_gbt = cross_val_predict(best_gbt, X_train_arr, y_train_arr, cv=inner_k, method="predict", n_jobs=1)
    oof_svr = cross_val_predict(best_svr, X_train_arr, y_train_arr, cv=inner_k, method="predict", n_jobs=1)

    meta_X_train = np.vstack([oof_gbt, oof_svr]).T
    meta = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta.fit(meta_X_train, y_train_arr)

    # -------------------------
    # Evaluate on the outer validation set
    # -------------------------
    X_val_arr, y_val_arr = sanitize_features_and_target(X_val_df, y_val_ser)

    preds_gbt_val = best_gbt.predict(X_val_arr)
    preds_svr_val = best_svr.predict(X_val_arr)
    preds_meta_val = meta.predict(np.vstack([preds_gbt_val, preds_svr_val]).T)

    # append accumulators
    y_test = np.concatenate([y_test, y_val_ser.values])
    y_pred = np.concatenate([y_pred, preds_meta_val])

    # metrics
    def metrics(y_t, y_p):
        mse = mean_squared_error(y_t, y_p)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    m_gbt = metrics(y_val_arr, preds_gbt_val)
    m_svr = metrics(y_val_arr, preds_svr_val)
    m_meta = metrics(y_val_arr, preds_meta_val)

    print(f"Fold {fold_idx} — GBT : RMSE={m_gbt['rmse']:.6f}, MAE={m_gbt['mae']:.6f}, R2={m_gbt['r2']:.6f}")
    print(f"Fold {fold_idx} — SVR : RMSE={m_svr['rmse']:.6f}, MAE={m_svr['mae']:.6f}, R2={m_svr['r2']:.6f}")
    print(f"Fold {fold_idx} — META: RMSE={m_meta['rmse']:.6f}, MAE={m_meta['mae']:.6f}, R2={m_meta['r2']:.6f}")

    print("Meta coefficients:", getattr(meta, "coef_", None), "intercept:", getattr(meta, "intercept_", None))

    fold_results.append({
        "fold": fold_idx,
        "gbt": m_gbt,
        "svr": m_svr,
        "meta": m_meta,
        "meta_coef": getattr(meta, "coef_", None).tolist() if getattr(meta, "coef_", None) is not None else None
    })

    print(f"Time for fold {fold_idx}: {time.time() - t0:.1f}s")

# -------------------------
# Aggregated metrics and outputs
# -------------------------
if y_test.size == 0:
    raise RuntimeError("y_test is empty after outer CV — something went wrong in the loops.")

agg_mse = mean_squared_error(y_test, y_pred)
agg_rmse = float(np.sqrt(agg_mse))
agg_mae = float(mean_absolute_error(y_test, y_pred))
agg_r2 = float(r2_score(y_test, y_pred))

print("\n" + "#"*60)
print("Nested-CV aggregated results (stacked meta predictions):")
print(f"Aggregated RMSE: {agg_rmse:.6f}")
print(f"Aggregated MAE : {agg_mae:.6f}")
print(f"Aggregated R2  : {agg_r2:.6f}")
print(f"Total nested-CV time: {time.time() - start_time:.1f}s")
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
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.6, s=20)
minv = float(min(np.min(y_test), np.min(y_pred)))
maxv = float(max(np.max(y_test), np.max(y_pred)))
ax.plot([minv, maxv], [minv, maxv], color="k", linestyle="--", linewidth=1)
ax.set_xlabel("True (y_test)")
ax.set_ylabel("Predicted (y_pred)")
ax.set_title(f"Pred vs True (RMSE={agg_rmse:.4f})")

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
    pass

# -------------------------
# FINAL MODEL (optional)
# -------------------------
if REFIT_FINAL:
    print("\nRefitting tuned models on full dataset (this may be slow)...")
    t_ref = time.time()
    FINAL_N_ITER = max(n_iter_bayes, 40)

    bayes_gbt_full = make_bayes_gbt(n_iter=FINAL_N_ITER)
    bayes_svr_full = make_bayes_svr(n_iter=FINAL_N_ITER)

    # sanitize full dataset
    X_full_arr, y_full_arr = sanitize_features_and_target(X, y)
    bayes_gbt_full.fit(X_full_arr, y_full_arr)
    try:
        bayes_svr_full.fit(X_full_arr, y_full_arr)
        best_svr_full = bayes_svr_full.best_estimator_
    except Exception as e:
        print("Warning: final SVR BayesSearch failed on full data:", repr(e))
        # fallback safe SVR pipeline
        safe_svr = SVR(C=1.0, kernel="rbf", gamma="scale", epsilon=0.1, max_iter=1000000)
        best_svr_full = Pipeline([("scaler", StandardScaler()), ("svr", safe_svr)])
        best_svr_full.fit(X_full_arr, y_full_arr)

    best_gbt_full = bayes_gbt_full.best_estimator_
    print("Final best GBT params:", getattr(bayes_gbt_full, "best_params_", None))
    if hasattr(bayes_svr_full, "best_params_"):
        print("Final best SVR params:", getattr(bayes_svr_full, "best_params_", None))

    # OOF preds across full dataset
    oof_gbt_full = cross_val_predict(best_gbt_full, X_full_arr, y_full_arr, cv=inner_k, method="predict", n_jobs=1)
    oof_svr_full = cross_val_predict(best_svr_full, X_full_arr, y_full_arr, cv=inner_k, method="predict", n_jobs=1)

    meta_X_full = np.vstack([oof_gbt_full, oof_svr_full]).T
    meta_final = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta_final.fit(meta_X_full, y_full_arr)

    joblib.dump(best_gbt_full, os.path.join("models", "best_gbt_full.joblib"))
    joblib.dump(best_svr_full, os.path.join("models", "best_svr_full.joblib"))
    joblib.dump(meta_final, os.path.join("models", "meta_final.joblib"))
    print(f"Final models saved to 'models/' (time: {time.time() - t_ref:.1f}s)")

print("Script finished.")
