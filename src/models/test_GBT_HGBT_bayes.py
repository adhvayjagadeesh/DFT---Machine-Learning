"""
GBT_HGBT_bayes.py

Nested CV + BayesSearchCV tuning for:
 - GradientBoostingRegressor (GBT)
 - HistGradientBoostingRegressor (HGBT)

Improvements vs original:
 - Outer KFold for honest evaluation, inner BayesSearchCV for tuning
 - Out-of-fold (OOF) stacking with a Ridge meta-learner (learned blend)
 - Safe single-threaded OOF predictions to avoid multiprocessing issues
 - Robust RMSE via sqrt(MSE)
 - Per-fold + aggregated metrics, predictions CSV, diagnostic plot, optional final refit and model saving
"""

import os
import time
import numpy as np
import multiprocessing
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sys

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Import dataset variables (x_, y_, k_) from your final.py
from data.final import x_, y_, k_

# -------------------------
# Configuration / safety
# -------------------------
RNG_SEED = 67
np.random.seed(RNG_SEED)

CPU_COUNT = max(1, multiprocessing.cpu_count())
# Safe defaults: increase only if you know your environment supports joblib/multiprocess reliably.
PARALLEL_PRED_JOBS = 1
BAYES_N_JOBS = 1

# Print versions for diagnostics (optional)
try:
    import sklearn
    print("sklearn:", sklearn.__version__)
except Exception:
    print("Could not determine sklearn version", file=sys.stderr)

# -------------------------
# Pipelines
# -------------------------
pipe_gbt = Pipeline([
    ("scaler", StandardScaler()),
    ("gbt", GradientBoostingRegressor(random_state=RNG_SEED))
])

pipe_hgbt = Pipeline([
    ("scaler", StandardScaler()),
    ("hgbt", HistGradientBoostingRegressor(random_state=RNG_SEED))
])

# -------------------------
# Hyperparameter spaces (UNCHANGED from your original)
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

hyperparams_hgbt = {
    "hgbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
    "hgbt__max_iter": Integer(150, 800),
    "hgbt__max_leaf_nodes": Integer(20, 50),
    "hgbt__min_samples_leaf": Integer(10, 40),
    "hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
    "hgbt__max_bins": Integer(127, 255),
}

# -------------------------
# CV / BayesSearch settings
# -------------------------
outer_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
inner_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
n_iter_bayes = 20
verbose_bayes = 2

def make_bayes_gbt(n_iter=n_iter_bayes):
    return BayesSearchCV(
        pipe_gbt,
        hyperparams_gbt,
        cv=inner_k,
        n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=verbose_bayes,
        random_state=RNG_SEED
    )

def make_bayes_hgbt(n_iter=n_iter_bayes):
    return BayesSearchCV(
        pipe_hgbt,
        hyperparams_hgbt,
        cv=inner_k,
        n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=verbose_bayes,
        random_state=RNG_SEED
    )

# -------------------------
# Prepare data & containers
# -------------------------
# Use full dataset variables exported by final.py
X = x_.copy().reset_index(drop=True)
y = y_.copy().reset_index(drop=True)

outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=RNG_SEED)

# Keep original-style accumulators (same semantics as your original script)
y_pred = np.array([])   # concatenated hybrid predictions across outer folds
y_test = np.array([])   # concatenated ground-truth across outer folds

fold_results = []
start_time = time.time()

print(f"Starting nested CV: outer_k={outer_k}, inner_k={inner_k}, n_iter_bayes={n_iter_bayes}")

# -------------------------
# Outer CV loop (nested workflow)
# -------------------------
for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
    t0 = time.time()
    print("\n" + "="*60)
    print(f"OUTER FOLD {fold_idx}/{outer_k}")
    print("="*60)

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # -------------------------
    # Inner tuning: GBT
    # -------------------------
    print("Tuning GradientBoosting (inner CV)...")
    bayes_gbt = make_bayes_gbt()
    bayes_gbt.fit(X_train, y_train)
    best_gbt = bayes_gbt.best_estimator_
    print(" -> Best GBT params:", bayes_gbt.best_params_)

    # -------------------------
    # Inner tuning: HGBT
    # -------------------------
    print("Tuning HistGradientBoosting (inner CV)...")
    bayes_hgbt = make_bayes_hgbt()
    bayes_hgbt.fit(X_train, y_train)
    best_hgbt = bayes_hgbt.best_estimator_
    print(" -> Best HGBT params:", bayes_hgbt.best_params_)

    # -------------------------
    # Create OOF predictions on the training partition for meta training
    # -------------------------
    print("Generating out-of-fold (OOF) predictions for meta training (safe single-threaded)...")
    # cross_val_predict clones the tuned estimators (keeps tuned hyperparams)
    oof_gbt = cross_val_predict(best_gbt, X_train, y_train, cv=inner_k, method="predict", n_jobs=1)
    oof_hgbt = cross_val_predict(best_hgbt, X_train, y_train, cv=inner_k, method="predict", n_jobs=1)

    meta_X_train = np.vstack([oof_gbt, oof_hgbt]).T
    meta = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta.fit(meta_X_train, y_train)

    # -------------------------
    # Evaluate on the outer validation set
    # -------------------------
    preds_gbt_val = best_gbt.predict(X_val)
    preds_hgbt_val = best_hgbt.predict(X_val)

    # Use learned meta-learner to blend predictions (preferred over fixed average)
    meta_X_val = np.vstack([preds_gbt_val, preds_hgbt_val]).T
    preds_meta_val = meta.predict(meta_X_val)

    # Append to accumulators (exact semantics as your original loop)
    y_test = np.concatenate([y_test, y_val.values])
    y_pred = np.concatenate([y_pred, preds_meta_val])

    # -------------------------
    # Compute metrics (robust RMSE)
    # -------------------------
    def metrics(y_t, y_p):
        y_t = np.asarray(y_t)
        y_p = np.asarray(y_p)
        mse = mean_squared_error(y_t, y_p)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    m_gbt = metrics(y_val, preds_gbt_val)
    m_hgbt = metrics(y_val, preds_hgbt_val)
    m_meta = metrics(y_val, preds_meta_val)

    print(f"Fold {fold_idx} — GBT : RMSE={m_gbt['rmse']:.6f}, MAE={m_gbt['mae']:.6f}, R2={m_gbt['r2']:.6f}")
    print(f"Fold {fold_idx} — HGBT: RMSE={m_hgbt['rmse']:.6f}, MAE={m_hgbt['mae']:.6f}, R2={m_hgbt['r2']:.6f}")
    print(f"Fold {fold_idx} — META: RMSE={m_meta['rmse']:.6f}, MAE={m_meta['mae']:.6f}, R2={m_meta['r2']:.6f}")

    # Print meta coefficients
    try:
        print("Meta weights (coef):", getattr(meta, "coef_", None), " intercept:", getattr(meta, "intercept_", None))
    except Exception:
        pass

    fold_results.append({
        "fold": fold_idx,
        "gbt": m_gbt,
        "hgbt": m_hgbt,
        "meta": m_meta,
        "meta_coef": getattr(meta, "coef_", None).tolist() if getattr(meta, "coef_", None) is not None else None
    })

    print(f"Time for fold {fold_idx}: {time.time() - t0:.1f}s")

# -------------------------
# Aggregated metrics
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
print(f"Total time (s) : {time.time() - start_time:.1f}")
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
# Plot predicted vs true + residuals
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
    # headless environments may fail on plt.show()
    pass

# -------------------------
# Optional final refit on full dataset & save tuned models
# -------------------------
REFIT_FINAL = True
FINAL_N_ITER = max(n_iter_bayes, 40)

if REFIT_FINAL:
    print("\nRefitting tuned models on full dataset (may be slow)...")
    t_ref = time.time()

    # Re-run BayesSearchCV on full data with possibly more iterations
    bayes_gbt_full = make_bayes_gbt(n_iter=FINAL_N_ITER)
    bayes_hgbt_full = make_bayes_hgbt(n_iter=FINAL_N_ITER)

    bayes_gbt_full.fit(X, y)
    bayes_hgbt_full.fit(X, y)

    best_gbt_full = bayes_gbt_full.best_estimator_
    best_hgbt_full = bayes_hgbt_full.best_estimator_
    print("Final best GBT params:", bayes_gbt_full.best_params_)
    print("Final best HGBT params:", bayes_hgbt_full.best_params_)

    # OOF preds across full dataset (for meta training)
    oof_gbt_full = cross_val_predict(best_gbt_full, X, y, cv=inner_k, method="predict", n_jobs=1)
    oof_hgbt_full = cross_val_predict(best_hgbt_full, X, y, cv=inner_k, method="predict", n_jobs=1)

    meta_X_full = np.vstack([oof_gbt_full, oof_hgbt_full]).T
    meta_final = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta_final.fit(meta_X_full, y)

    # Save final artifacts
    joblib.dump(best_gbt_full, os.path.join("models", "best_gbt_full.joblib"))
    joblib.dump(best_hgbt_full, os.path.join("models", "best_hgbt_full.joblib"))
    joblib.dump(meta_final, os.path.join("models", "meta_final.joblib"))
    print(f"Saved final models to models/ (time: {time.time()-t_ref:.1f}s)")

print("Script finished.")