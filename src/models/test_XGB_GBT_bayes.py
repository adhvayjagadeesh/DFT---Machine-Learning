"""
XGB_GBT_bayes.py

Nested CV + BayesSearchCV tuning for XGBRegressor and GradientBoostingRegressor,
OOF stacking (Ridge) to learn ensemble weights, with outputs:
 - per-fold and aggregated metrics (RMSE, MAE, R2)
 - saved CSV of predictions models/predictions.csv
 - saved figure models/pred_vs_true.png
 - optional final refit & saved models in models/
 
Only XGB and GBT used — NO RandomForest anywhere.
"""

import os
import time
import numpy as np
import multiprocessing
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# dataset variables from final.py
from data.final import x_, y_, k_

# -------------------------
# Configuration / safety
# -------------------------
RNG_SEED = 67
np.random.seed(RNG_SEED)

CPU_COUNT = max(1, multiprocessing.cpu_count())
PARALLEL_PRED_JOBS = 1   # safe default
BAYES_N_JOBS = 1     # safe default

# -------------------------
# Pipelines (scaling + estimator)
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

pipe_gbt = Pipeline([
  ("scaler", StandardScaler()),
  ("gbt", GradientBoostingRegressor(random_state=RNG_SEED))
])

# -------------------------
# Hyperparameter spaces
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
}

hyperparams_gbt = {
  "gbt__n_estimators": Integer(100, 500),
  "gbt__learning_rate": Real(1e-3, 1e-1, prior="log-uniform"),
  "gbt__max_depth": Integer(2, 8),
  "gbt__min_samples_split": Integer(2, 12),
  "gbt__min_samples_leaf": Integer(1, 6),
  "gbt__subsample": Real(0.6, 1.0),
  "gbt__max_features": Categorical(["sqrt", "log2", None]),
}

# -------------------------
# CV / search params
# -------------------------
outer_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
inner_k = k_ if (isinstance(k_, int) and k_ >= 2) else 4
n_iter_bayes = 20
verbose_bayes = 2

def make_bayes_xgb(n_iter=n_iter_bayes):
  return BayesSearchCV(
    pipe_xgb,
    hyperparams_xgb,
    cv=inner_k,
    n_iter=n_iter,
    n_jobs=BAYES_N_JOBS,
    scoring="neg_mean_squared_error",
    verbose=verbose_bayes,
    random_state=RNG_SEED
  )

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

# -------------------------
# Prepare data and accumulators
# -------------------------
X = x_.copy().reset_index(drop=True)
y = y_.copy().reset_index(drop=True)

outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=RNG_SEED)

# keep original-style accumulators
y_pred = np.array([])
y_test = np.array([])

fold_results = []
t_global_start = time.time()
print(f"Starting nested CV: outer_k={outer_k}, inner_k={inner_k}, n_iter_bayes={n_iter_bayes}")

# -------------------------
# Outer CV loop (nested flow)
# -------------------------
for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
  t0 = time.time()
  print("\n" + "="*64)
  print(f"OUTER FOLD {fold_idx}/{outer_k}")
  print("="*64)

  X_train = X.iloc[train_idx]
  y_train = y.iloc[train_idx]
  X_val = X.iloc[val_idx]
  y_val = y.iloc[val_idx]

  # ---- inner tuning: XGB ----
  print("Tuning XGBoost (inner CV)...")
  bayes_xgb = make_bayes_xgb()
  bayes_xgb.fit(X_train, y_train)
  best_xgb = bayes_xgb.best_estimator_
  print(" -> Best XGB params:", bayes_xgb.best_params_)

  # ---- inner tuning: GBT ----
  print("Tuning GradientBoosting (inner CV)...")
  bayes_gbt = make_bayes_gbt()
  bayes_gbt.fit(X_train, y_train)
  best_gbt = bayes_gbt.best_estimator_
  print(" -> Best GBT params:", bayes_gbt.best_params_)

  # ---- OOF preds for meta training (single-threaded safe) ----
  print("Generating OOF predictions for meta training...")
  oof_xgb = cross_val_predict(best_xgb, X_train, y_train, cv=inner_k, method="predict", n_jobs=1)
  oof_gbt = cross_val_predict(best_gbt, X_train, y_train, cv=inner_k, method="predict", n_jobs=1)

  meta_X_train = np.vstack([oof_xgb, oof_gbt]).T
  meta = Ridge(alpha=1.0, random_state=RNG_SEED)
  meta.fit(meta_X_train, y_train)

  # ---- Evaluate on outer validation set ----
  preds_xgb = best_xgb.predict(X_val)
  preds_gbt = best_gbt.predict(X_val)

  # blend via learned meta (recommended)
  preds_blend = meta.predict(np.vstack([preds_xgb, preds_gbt]).T)

  # maintain original accumulators
  y_test = np.concatenate([y_test, y_val.values])
  y_pred = np.concatenate([y_pred, preds_blend])

  # metrics (robust RMSE calculation)
  def metrics(y_t, y_p):
    y_t = np.asarray(y_t)
    y_p = np.asarray(y_p)
    mse = mean_squared_error(y_t, y_p)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    return {"rmse": rmse, "mae": mae, "r2": r2}

  m_xgb = metrics(y_val, preds_xgb)
  m_gbt = metrics(y_val, preds_gbt)
  m_meta = metrics(y_val, preds_blend)

  print(f"Fold {fold_idx} — XGB: RMSE={m_xgb['rmse']:.6f}, MAE={m_xgb['mae']:.6f}, R2={m_xgb['r2']:.6f}")
  print(f"Fold {fold_idx} — GBT: RMSE={m_gbt['rmse']:.6f}, MAE={m_gbt['mae']:.6f}, R2={m_gbt['r2']:.6f}")
  print(f"Fold {fold_idx} — META: RMSE={m_meta['rmse']:.6f}, MAE={m_meta['mae']:.6f}, R2={m_meta['r2']:.6f}")

  # meta coefficients for interpretation
  print("Meta coefficients:", getattr(meta, "coef_", None), " intercept:", getattr(meta, "intercept_", None))

  fold_results.append({
    "fold": fold_idx,
    "xgb": m_xgb,
    "gbt": m_gbt,
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

print("\n" + "#"*80)
print("Nested-CV aggregated results:")
print(f"Aggregated RMSE: {agg_rmse:.6f}")
print(f"Aggregated MAE : {agg_mae:.6f}")
print(f"Aggregated R2  : {agg_r2:.6f}")
print("#"*80)

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
  pass

# -------------------------
# Optional final refit on full dataset & save tuned models
# -------------------------
REFIT_FINAL = True
FINAL_N_ITER = max(n_iter_bayes, 40)

if REFIT_FINAL:
  print("\nRefitting tuned models on full dataset (this can be slow)...")
  t_ref = time.time()

  bayes_xgb_full = make_bayes_xgb(n_iter=FINAL_N_ITER)
  bayes_gbt_full = make_bayes_gbt(n_iter=FINAL_N_ITER)

  bayes_xgb_full.fit(X, y)
  bayes_gbt_full.fit(X, y)

  best_xgb_full = bayes_xgb_full.best_estimator_
  best_gbt_full = bayes_gbt_full.best_estimator_
  print("Final best XGB params:", bayes_xgb_full.best_params_)
  print("Final best GBT params:", bayes_gbt_full.best_params_)

  oof_xgb_full = cross_val_predict(best_xgb_full, X, y, cv=inner_k, method="predict", n_jobs=1)
  oof_gbt_full = cross_val_predict(best_gbt_full, X, y, cv=inner_k, method="predict", n_jobs=1)

  meta_X_full = np.vstack([oof_xgb_full, oof_gbt_full]).T
  meta_final = Ridge(alpha=1.0, random_state=RNG_SEED)
  meta_final.fit(meta_X_full, y)

  joblib.dump(best_xgb_full, os.path.join("models", "best_xgb_full.joblib"))
  joblib.dump(best_gbt_full, os.path.join("models", "best_gbt_full.joblib"))
  joblib.dump(meta_final, os.path.join("models", "meta_final.joblib"))
  print(f"Saved final models to models/ (time: {time.time()-t_ref:.1f}s)")

print("Script finished.")