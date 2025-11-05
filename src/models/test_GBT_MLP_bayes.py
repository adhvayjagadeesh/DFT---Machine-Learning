"""
hybrid_gbt_mlp_bayes_robust.py

Robust hybrid: GradientBoostingRegressor (GBT) + MLP (wrapped) with:
 - nested CV (outer KFold = k_, inner BayesSearchCV)
 - OOF stacking (Ridge meta-learner)
 - sanitization of input (replace inf, fill NaN)
 - safe MLP wrapper so skopt can search integer hidden sizes
 - constrained MLP hyperparams (learning rate, alpha) and early stopping
 - fault-tolerant MLP BayesSearchCV (fallback default if tuning crashes)
 - saves models/predictions.csv and models/pred_vs_true.png
"""

import os
import time
import sys
import numpy as np
import pandas as pd
import multiprocessing
import joblib
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# import dataset variables (x_, y_, k_) from final.py
from data.final import x_, y_, k_

# -------------------------
# MLP wrapper (robust)
# -------------------------
class MLPWrapper(BaseEstimator, RegressorMixin):
  """
  Wraps MLPRegressor while exposing integer hidden_layer_size and alpha for skopt.
  Uses early_stopping to prevent runaway divergence.
  """
  def __init__(self,
         hidden_layer_size=200,
         solver="adam",
         learning_rate_init=1e-3,
         max_iter=300,
         alpha=1e-4,
         early_stopping=True,
         validation_fraction=0.1,
         n_iter_no_change=10,
         tol=1e-4,
         random_state=None):
    self.hidden_layer_size = int(hidden_layer_size)
    self.solver = solver
    self.learning_rate_init = float(learning_rate_init)
    self.max_iter = int(max_iter)
    self.alpha = float(alpha)
    self.early_stopping = bool(early_stopping)
    self.validation_fraction = float(validation_fraction)
    self.n_iter_no_change = int(n_iter_no_change)
    self.tol = float(tol)
    self.random_state = random_state
    self._mlp = None

  def fit(self, X, y):
    # Defensive: ensure arrays are float
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Build MLPRegressor with current params
    self._mlp = MLPRegressor(
      hidden_layer_sizes=(int(self.hidden_layer_size),),
      solver=self.solver,
      learning_rate_init=float(self.learning_rate_init),
      max_iter=int(self.max_iter),
      alpha=float(self.alpha),
      early_stopping=self.early_stopping,
      validation_fraction=float(self.validation_fraction),
      n_iter_no_change=int(self.n_iter_no_change),
      tol=float(self.tol),
      random_state=self.random_state,
      verbose=False
    )
    self._mlp.fit(X, y)
    return self

  def predict(self, X):
    return self._mlp.predict(X)

  def get_params(self, deep=True):
    return {
      "hidden_layer_size": self.hidden_layer_size,
      "solver": self.solver,
      "learning_rate_init": self.learning_rate_init,
      "max_iter": self.max_iter,
      "alpha": self.alpha,
      "early_stopping": self.early_stopping,
      "validation_fraction": self.validation_fraction,
      "n_iter_no_change": self.n_iter_no_change,
      "tol": self.tol,
      "random_state": self.random_state
    }

  def set_params(self, **params):
    for k, v in params.items():
      setattr(self, k, v)
    return self

# -------------------------
# Helpers: sanitize X/y
# -------------------------
def sanitize_features_and_target(X, y):
  """
  Replace +/-inf with NaN, fill NaN with column mean (for features),
  and assert target y has no NaN/inf.
  Returns numpy arrays (X_clean, y_clean).
  """
  # Accept pandas DataFrame or numpy array
  if isinstance(X, pd.DataFrame):
    Xc = X.copy()
    Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
    # fill NaNs with column means; if entire column is NaN, fill 0
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
    # where column entirely nan, set mean to 0
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(Xa))
    if inds[0].size > 0:
      Xa[inds] = np.take(col_means, inds[1])
    X_out = Xa

  y_out = np.asarray(y, dtype=float)
  # Check y for finiteness
  if not np.isfinite(y_out).all():
    # if NaNs present, raise useful error
    if np.isnan(y_out).any():
      raise ValueError("Target y contains NaN values. Clean target before training.")
    else:
      # replace +/-inf by large finite numbers? safer to raise.
      raise ValueError("Target y contains non-finite values (inf). Clean target before training.")
  return X_out, y_out

# -------------------------
# Config / safety
# -------------------------
RNG_SEED = 67
np.random.seed(RNG_SEED)
CPU_COUNT = max(1, multiprocessing.cpu_count())
PARALLEL_PRED_JOBS = 1
BAYES_N_JOBS = 1

# DEBUG toggle: set True for very fast smoke tests
DEBUG = False

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
# Pipelines
# -------------------------
pipe_gbt = Pipeline([
  ("scaler", StandardScaler()),
  ("gbt", GradientBoostingRegressor(random_state=RNG_SEED))
])

pipe_mlpw = Pipeline([
  ("scaler", StandardScaler()),
  ("mlpw", MLPWrapper(random_state=RNG_SEED))
])

# -------------------------
# Hyperparameter spaces
# GBT left exactly as requested
# MLP constrained: smaller learning rate upper bound and alpha regularization
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

hyperparams_mlp = {
  # wrapper exposes integer hidden_layer_size
  "mlpw__hidden_layer_size": Integer(100, 400),
  "mlpw__solver": Categorical(["adam", "sgd"]),
  # constrain lr to avoid explosive updates
  "mlpw__learning_rate_init": Real(1e-5, 1e-2, prior="log-uniform"),
  "mlpw__max_iter": Integer(150, 600),
  # regularization to stabilize training
  "mlpw__alpha": Real(1e-6, 1e-2, prior="log-uniform"),
  # you can also search n_iter_no_change or tol if desired
}

# -------------------------
# Bayes factories
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

def make_bayes_mlp(n_iter=n_iter_bayes):
  return BayesSearchCV(
    pipe_mlpw,
    hyperparams_mlp,
    cv=inner_k,
    n_iter=n_iter,
    n_jobs=BAYES_N_JOBS,
    scoring="neg_mean_squared_error",
    verbose=verbose_bayes,
    random_state=RNG_SEED,
    error_score=np.nan
  )

# (error_score added to tolerate candidate failures; older skopt may ignore it — fallback try/except implemented below)

# -------------------------
# Prepare data & accumulators
# -------------------------
X = x_.copy().reset_index(drop=True)
y = y_.copy().reset_index(drop=True)
outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=RNG_SEED)

y_pred = np.array([])
y_test = np.array([])

fold_results = []
start_time = time.time()

print(f"Starting nested CV (robust): outer_k={outer_k}, inner_k={inner_k}, n_iter_bayes={n_iter_bayes}")

# -------------------------
# Outer CV loop
# -------------------------
for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
  t0 = time.time()
  print("\n" + "="*60)
  print(f"OUTER FOLD {fold_idx}/{outer_k}")
  print("="*60)

  X_train_df, X_val_df = X.iloc[train_idx], X.iloc[val_idx]
  y_train_ser, y_val_ser = y.iloc[train_idx], y.iloc[val_idx]

  # sanitize the training partition (features & target)
  X_train_arr, y_train_arr = sanitize_features_and_target(X_train_df, y_train_ser)

  # Note: BayesSearchCV / pipeline expects 2D X; we pass sanitized numpy arrays.
  # For the validation set we keep DataFrame/Series to allow concatenation later, but we sanitize before predictions where needed.
  # Tune GBT (safe)
  print("Tuning GBT (inner CV)...")
  bayes_gbt = make_bayes_gbt()
  bayes_gbt.fit(X_train_arr, y_train_arr)  # pipeline contains scaler, so pass raw numeric arrays
  best_gbt = bayes_gbt.best_estimator_
  print(" -> Best GBT params:", bayes_gbt.best_params_)

  # Tune MLP with robust error handling
  print("Tuning MLP (inner CV) — robust try/except ...")
  bayes_mlp = make_bayes_mlp()
  try:
    bayes_mlp.fit(X_train_arr, y_train_arr)
    best_mlpw = bayes_mlp.best_estimator_
    print(" -> Best MLP wrapper params:", bayes_mlp.best_params_)
  except Exception as e:
    # log and fallback to a safe default MLP if BayesSearch fails
    print("Warning: MLP BayesSearchCV failed for this fold with exception:", repr(e))
    print("Falling back to safe default MLP wrapper (no tuning) for this fold.")
    safe_mlp = MLPWrapper(
      hidden_layer_size=100,
      solver="adam",
      learning_rate_init=1e-3,
      max_iter=300,
      alpha=1e-4,
      early_stopping=True,
      n_iter_no_change=10,
      tol=1e-4,
      random_state=RNG_SEED
    )
    # We need a pipeline consistent with the rest
    best_mlpw = Pipeline([
      ("scaler", StandardScaler()),
      ("mlpw", safe_mlp)
    ])
    # fit the fallback on sanitized training data
    best_mlpw.fit(X_train_arr, y_train_arr)

  # -------------------------
  # OOF predictions for meta
  # -------------------------
  print("Generating OOF predictions for meta training (single-threaded safe)...")
  # cross_val_predict clones the tuned estimators/pipelines; pass arrays for X
  # n_jobs=1 to avoid nested parallelism
  try:
    oof_gbt = cross_val_predict(best_gbt, X_train_arr, y_train_arr, cv=inner_k, method="predict", n_jobs=1)
  except Exception as e:
    print("Warning: cross_val_predict failed for GBT on OOF; trying single-threaded fallback:", repr(e))
    oof_gbt = cross_val_predict(best_gbt, X_train_arr, y_train_arr, cv=inner_k, method="predict", n_jobs=1)

  try:
    oof_mlp = cross_val_predict(best_mlpw, X_train_arr, y_train_arr, cv=inner_k, method="predict", n_jobs=1)
  except Exception as e:
    print("Warning: cross_val_predict failed for MLP on OOF; trying single-threaded fallback:", repr(e))
    oof_mlp = cross_val_predict(best_mlpw, X_train_arr, y_train_arr, cv=inner_k, method="predict", n_jobs=1)

  meta_X_train = np.vstack([oof_gbt, oof_mlp]).T
  meta = Ridge(alpha=1.0, random_state=RNG_SEED)
  meta.fit(meta_X_train, y_train_arr)

  # -------------------------
  # Evaluate on the outer validation set
  # -------------------------
  # sanitize X_val for prediction (but keep original y_val_ser for accumulator)
  X_val_arr, y_val_arr = sanitize_features_and_target(X_val_df, y_val_ser)

  preds_gbt_val = best_gbt.predict(X_val_arr)
  preds_mlp_val = best_mlpw.predict(X_val_arr) if isinstance(best_mlpw, Pipeline) else best_mlpw.predict(X_val_arr)
  preds_meta_val = meta.predict(np.vstack([preds_gbt_val, preds_mlp_val]).T)

  # accumulate results (y_val_ser is a pandas Series; use its .values)
  y_test = np.concatenate([y_test, y_val_ser.values])
  y_pred = np.concatenate([y_pred, preds_meta_val])

  # metrics
  def metrics(y_t, y_p):
    mse = mean_squared_error(y_t, y_p)
    return {"rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_t, y_p)),
        "r2": float(r2_score(y_t, y_p))}

  m_gbt = metrics(y_val_ser.values, preds_gbt_val)
  m_mlp = metrics(y_val_ser.values, preds_mlp_val)
  m_meta = metrics(y_val_ser.values, preds_meta_val)

  print(f"Fold {fold_idx} — GBT : RMSE={m_gbt['rmse']:.6f}, MAE={m_gbt['mae']:.6f}, R2={m_gbt['r2']:.6f}")
  print(f"Fold {fold_idx} — MLP : RMSE={m_mlp['rmse']:.6f}, MAE={m_mlp['mae']:.6f}, R2={m_mlp['r2']:.6f}")
  print(f"Fold {fold_idx} — META: RMSE={m_meta['rmse']:.6f}, MAE={m_meta['mae']:.6f}, R2={m_meta['r2']:.6f}")

  print("Meta coefficients:", getattr(meta, "coef_", None), "intercept:", getattr(meta, "intercept_", None))

  fold_results.append({
    "fold": fold_idx,
    "gbt": m_gbt, "mlp": m_mlp, "meta": m_meta,
    "meta_coef": getattr(meta, "coef_", None).tolist() if getattr(meta, "coef_", None) is not None else None
  })

  print(f"Time for fold {fold_idx}: {time.time() - t0:.1f}s")

# -------------------------
# Aggregated metrics & outputs
# -------------------------
if y_test.size == 0:
  raise RuntimeError("y_test is empty after outer CV — something went wrong.")

agg_mse = mean_squared_error(y_test, y_pred)
agg_rmse = float(np.sqrt(agg_mse))
agg_mae = float(mean_absolute_error(y_test, y_pred))
agg_r2 = float(r2_score(y_test, y_pred))

print("\n" + "#"*80)
print("Nested-CV aggregated results:")
print(f"Aggregated RMSE: {agg_rmse:.6f}")
print(f"Aggregated MAE : {agg_mae:.6f}")
print(f"Aggregated R2  : {agg_r2:.6f}")
print(f"Total time (s) : {time.time() - start_time:.1f}")
print("#"*80)

# Save predictions CSV
os.makedirs("models", exist_ok=True)
pred_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
pred_df["residual"] = pred_df["y_test"] - pred_df["y_pred"]
pred_csv_path = os.path.join("models", "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"Saved predictions CSV -> {pred_csv_path}")

# Plot preds vs true + residuals
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
# Optional final refit & save tuned models
# -------------------------
if REFIT_FINAL:
  print("\nRefitting tuned models on full dataset (may be slow)...")
  t_ref = time.time()
  FINAL_N_ITER = max(n_iter_bayes, 40)

  bayes_gbt_full = make_bayes_gbt(n_iter=FINAL_N_ITER)
  bayes_mlp_full = make_bayes_mlp(n_iter=FINAL_N_ITER)

  # sanitize full dataset before fitting
  X_full_arr, y_full_arr = sanitize_features_and_target(X, y)
  bayes_gbt_full.fit(X_full_arr, y_full_arr)
  try:
    bayes_mlp_full.fit(X_full_arr, y_full_arr)
  except Exception as e:
    print("Warning: final MLP BayesSearch failed on full data:", repr(e))
    # fallback: fit a safe mlp on full data
    safe_mlp = MLPWrapper(hidden_layer_size=100, learning_rate_init=1e-3, max_iter=300, alpha=1e-4, random_state=RNG_SEED)
    safe_pipe = Pipeline([("scaler", StandardScaler()), ("mlpw", safe_mlp)])
    safe_pipe.fit(X_full_arr, y_full_arr)
    best_mlp_full = safe_pipe
  else:
    best_mlp_full = bayes_mlp_full.best_estimator_

  best_gbt_full = bayes_gbt_full.best_estimator_
  print("Final best GBT params:", bayes_gbt_full.best_params_)
  if hasattr(bayes_mlp_full, "best_params_"):
    print("Final best MLP params:", bayes_mlp_full.best_params_)

  # OOF preds for meta on full data
  oof_gbt_full = cross_val_predict(best_gbt_full, X_full_arr, y_full_arr, cv=inner_k, method="predict", n_jobs=1)
  oof_mlp_full = cross_val_predict(best_mlp_full, X_full_arr, y_full_arr, cv=inner_k, method="predict", n_jobs=1)

  meta_X_full = np.vstack([oof_gbt_full, oof_mlp_full]).T
  meta_final = Ridge(alpha=1.0, random_state=RNG_SEED)
  meta_final.fit(meta_X_full, y_full_arr)

  joblib.dump(best_gbt_full, os.path.join("models", "best_gbt_full.joblib"))
  joblib.dump(best_mlp_full, os.path.join("models", "best_mlp_full.joblib"))
  joblib.dump(meta_final, os.path.join("models", "meta_final.joblib"))
  print(f"Saved final models to models/ (time: {time.time()-t_ref:.1f}s)")

print("Script finished.")