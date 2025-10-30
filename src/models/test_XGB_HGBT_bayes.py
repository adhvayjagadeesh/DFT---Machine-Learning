# improved_xgb_hgbt_bayes.py
"""
Nested CV + BayesSearchCV tuning for XGBRegressor and HistGradientBoostingRegressor,
OOF stacking (Ridge) to learn ensemble weights, with outputs:
 - per-fold and aggregated metrics (RMSE, MAE, R2)
 - saved CSV of predictions models/predictions.csv
 - saved figure models/pred_vs_true.png
 - optional final refit & saved models in models/
 
Only XGB and HGBT used — NO RandomForest or other models.
"""

import os
import time
import multiprocessing
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

# dataset variables from final.py
from data.final import x_ as X_RAW, y_ as y_RAW, k_ as k_from_data

# -------------------------
# Config / safe parallelism
# -------------------------
RNG_SEED = 67
np.random.seed(RNG_SEED)

CPU_COUNT = max(1, multiprocessing.cpu_count())
PARALLEL_PRED_JOBS = 1    # safe default for cross_val_predict
BAYES_N_JOBS = 1          # safe default for BayesSearchCV parallel trials
N_ITER_BAYES = 20         # per-fold tuning iterations (moderate)
FINAL_N_ITER = 60         # for final refit (toggleable)
REFIT_FINAL = True        # set False to skip heavy final refit
VERBOSE_BAYES = 2

os.makedirs("models", exist_ok=True)

print("Safe config: CPU_COUNT =", CPU_COUNT, "PARALLEL_PRED_JOBS =", PARALLEL_PRED_JOBS, "BAYES_N_JOBS =", BAYES_N_JOBS)

# -------------------------
# Prepare data
# -------------------------
X = X_RAW.copy().reset_index(drop=True)
y = y_RAW.copy().reset_index(drop=True)

outer_k = int(k_from_data) if (isinstance(k_from_data, int) and k_from_data >= 2) else 4
# prefer inner_k <= outer_k-1 and reasonable value (2..5)
inner_k = min(5, max(2, outer_k - 1))

outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=RNG_SEED)

print(f"Outer folds: {outer_k}, Inner folds: {inner_k}")

# -------------------------
# Pipelines (deterministic XGB + HGBT)
# -------------------------
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

pipe_hgbt = Pipeline([
    ("scaler", StandardScaler()),
    ("hgbt", HistGradientBoostingRegressor(random_state=RNG_SEED))
])

# -------------------------
# Narrowed safer hyperparameter spaces
# -------------------------
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
    # removed tree_method from search (we set hist above)
}

hyperparams_hgbt = {
    "hgbt__learning_rate": Real(1e-3, 1e-1, prior="log-uniform"),
    "hgbt__max_iter": Integer(100, 500),
    "hgbt__max_leaf_nodes": Integer(10, 50),
    "hgbt__min_samples_leaf": Integer(1, 10),
    "hgbt__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
    "hgbt__max_bins": Integer(64, 255),
}

# -------------------------
# BayesSearch factories (safe n_jobs)
# -------------------------
def make_bayes_xgb(n_iter=N_ITER_BAYES):
    return BayesSearchCV(
        pipe_xgb,
        hyperparams_xgb,
        cv=inner_k,
        n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=VERBOSE_BAYES,
        random_state=RNG_SEED,
        refit=True
    )

def make_bayes_hgbt(n_iter=N_ITER_BAYES):
    return BayesSearchCV(
        pipe_hgbt,
        hyperparams_hgbt,
        cv=inner_k,
        n_iter=n_iter,
        n_jobs=BAYES_N_JOBS,
        scoring="neg_mean_squared_error",
        verbose=VERBOSE_BAYES,
        random_state=RNG_SEED,
        refit=True
    )

# -------------------------
# Accumulators & tracking
# -------------------------
y_pred = np.array([])
y_test = np.array([])
fold_results = []
t_global_start = time.time()

print("Starting nested CV tuning & OOF stacking...")

# -------------------------
# Outer loop: nested CV
# -------------------------
for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
    t0 = time.time()
    print("\n" + "=" * 60)
    print(f"OUTER FOLD {fold_idx}/{outer_k}")
    print("=" * 60)

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Tune XGB
    print("Tuning XGB (inner CV)...")
    bayes_xgb = make_bayes_xgb()
    bayes_xgb.fit(X_train, y_train)
    best_xgb = bayes_xgb.best_estimator_
    print(" -> Best XGB params:", bayes_xgb.best_params_)

    # Tune HGBT
    print("Tuning HistGradientBoosting (inner CV)...")
    bayes_hgbt = make_bayes_hgbt()
    bayes_hgbt.fit(X_train, y_train)
    best_hgbt = bayes_hgbt.best_estimator_
    print(" -> Best HGBT params:", bayes_hgbt.best_params_)

    # OOF predictions for meta training
    print("Generating OOF predictions for meta training...")
    try:
        oof_xgb = cross_val_predict(best_xgb, X_train, y_train, cv=inner_k, method="predict", n_jobs=PARALLEL_PRED_JOBS)
        oof_hgbt = cross_val_predict(best_hgbt, X_train, y_train, cv=inner_k, method="predict", n_jobs=PARALLEL_PRED_JOBS)
    except Exception as e:
        print("OOF parallel failed, falling back to single-threaded cross_val_predict:", e)
        oof_xgb = cross_val_predict(best_xgb, X_train, y_train, cv=inner_k, method="predict", n_jobs=1)
        oof_hgbt = cross_val_predict(best_hgbt, X_train, y_train, cv=inner_k, method="predict", n_jobs=1)

    meta_X_train = np.vstack([oof_xgb, oof_hgbt]).T
    meta = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta.fit(meta_X_train, y_train)

    # Predictions on outer validation
    preds_xgb = best_xgb.predict(X_val)
    preds_hgbt = best_hgbt.predict(X_val)
    preds_meta = meta.predict(np.vstack([preds_xgb, preds_hgbt]).T)

    # accumulate
    y_test = np.concatenate([y_test, y_val.values])
    y_pred = np.concatenate([y_pred, preds_meta])

    # compute metrics
    def metrics(y_t, y_p):
        mse = mean_squared_error(y_t, y_p)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_t, y_p)),
            "r2": float(r2_score(y_t, y_p))
        }

    m_xgb = metrics(y_val, preds_xgb)
    m_hgbt = metrics(y_val, preds_hgbt)
    m_meta = metrics(y_val, preds_meta)

    print(f"Fold {fold_idx} — XGB:  RMSE={m_xgb['rmse']:.6f}, MAE={m_xgb['mae']:.6f}, R2={m_xgb['r2']:.6f}")
    print(f"Fold {fold_idx} — HGBT: RMSE={m_hgbt['rmse']:.6f}, MAE={m_hgbt['mae']:.6f}, R2={m_hgbt['r2']:.6f}")
    print(f"Fold {fold_idx} — META: RMSE={m_meta['rmse']:.6f}, MAE={m_meta['mae']:.6f}, R2={m_meta['r2']:.6f}")

    print("Meta coefficients:", getattr(meta, "coef_", None), " intercept:", getattr(meta, "intercept_", None))

    fold_results.append({
        "fold": fold_idx,
        "xgb": m_xgb,
        "hgbt": m_hgbt,
        "meta": m_meta,
        "meta_coef": getattr(meta, "coef_", None).tolist() if getattr(meta, "coef_", None) is not None else None,
        "time_s": time.time() - t0
    })

    print(f"Fold time: {fold_results[-1]['time_s']:.1f}s")

# -------------------------
# Aggregated metrics
# -------------------------
if y_test.size == 0:
    raise RuntimeError("y_test is empty after outer CV — something went wrong in the loops.")

agg_mse = mean_squared_error(y_test, y_pred)
agg_rmse = float(np.sqrt(agg_mse))
agg_mae = float(mean_absolute_error(y_test, y_pred))
agg_r2 = float(r2_score(y_test, y_pred))

print("\n" + "#" * 80)
print("Nested-CV aggregated results:")
print(f"Aggregated RMSE: {agg_rmse:.6f}")
print(f"Aggregated MAE : {agg_mae:.6f}")
print(f"Aggregated R2  : {agg_r2:.6f}")
print("#" * 80)

# -------------------------
# Save predictions CSV
# -------------------------
pred_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
pred_df["residual"] = pred_df["y_test"] - pred_df["y_pred"]
pred_csv_path = os.path.join("models", "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"Saved predictions CSV -> {pred_csv_path}")

# -------------------------
# Plot predicted vs true + residuals
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(pred_df["y_test"], pred_df["y_pred"], alpha=0.6, s=20)
minv = float(min(pred_df["y_test"].min(), pred_df["y_pred"].min()))
maxv = float(max(pred_df["y_test"].max(), pred_df["y_pred"].max()))
axes[0].plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1)
axes[0].set_xlabel("True (y_test)")
axes[0].set_ylabel("Predicted (y_pred)")
axes[0].set_title(f"Pred vs True (RMSE={agg_rmse:.4f})")

axes[1].hist(pred_df["residual"].values, bins=40)
axes[1].set_title("Residuals histogram")
axes[1].set_xlabel("Residual (true - pred)")
axes[1].set_ylabel("Count")

plt.tight_layout()
fig_path = os.path.join("models", "pred_vs_true.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved figure -> {fig_path}")
try:
    plt.show()
except Exception:
    pass

# -------------------------
# Optional: final refit on full data and save tuned models
# -------------------------
if REFIT_FINAL:
    print("\nRefitting tuned models on full dataset (this can be slow)...")
    t_ref = time.time()

    bayes_xgb_full = make_bayes_xgb(n_iter=FINAL_N_ITER)
    bayes_hgbt_full = make_bayes_hgbt(n_iter=FINAL_N_ITER)

    bayes_xgb_full.fit(X, y)
    bayes_hgbt_full.fit(X, y)

    best_xgb_full = bayes_xgb_full.best_estimator_
    best_hgbt_full = bayes_hgbt_full.best_estimator_

    print("Final best XGB params:", bayes_xgb_full.best_params_)
    print("Final best HGBT params:", bayes_hgbt_full.best_params_)

    try:
        oof_xgb_full = cross_val_predict(best_xgb_full, X, y, cv=inner_k, method="predict", n_jobs=PARALLEL_PRED_JOBS)
        oof_hgbt_full = cross_val_predict(best_hgbt_full, X, y, cv=inner_k, method="predict", n_jobs=PARALLEL_PRED_JOBS)
    except Exception as e:
        print("Full-data OOF parallel failed, falling back to single-thread:", e)
        oof_xgb_full = cross_val_predict(best_xgb_full, X, y, cv=inner_k, method="predict", n_jobs=1)
        oof_hgbt_full = cross_val_predict(best_hgbt_full, X, y, cv=inner_k, method="predict", n_jobs=1)

    meta_X_full = np.vstack([oof_xgb_full, oof_hgbt_full]).T
    meta_final = Ridge(alpha=1.0, random_state=RNG_SEED)
    meta_final.fit(meta_X_full, y)

    joblib.dump(best_xgb_full, os.path.join("models", "best_xgb_full.joblib"))
    joblib.dump(best_hgbt_full, os.path.join("models", "best_hgbt_full.joblib"))
    joblib.dump(meta_final, os.path.join("models", "meta_final.joblib"))

    print(f"Saved final models to models/ (time: {time.time() - t_ref:.1f}s)")

print("Script finished.")