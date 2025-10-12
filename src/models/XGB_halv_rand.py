# idk y, but this bro is not deterministic, maybe it's cuz it's experimental

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from scipy.stats import randint, loguniform, uniform

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor())
])

hyperparams = {
    "xgb__max_depth": randint(3, 11),
    "xgb__min_child_weight": randint(1, 9),
    "xgb__learning_rate": loguniform(1e-2, 0.2),
    "xgb__n_estimators": randint(100, 801),
    "xgb__subsample": uniform(0.7, 0.3),
    "xgb__colsample_bytree": uniform(0.6, 0.4),
    "xgb__reg_alpha": loguniform(1e-5, 0.5),
    "xgb__reg_lambsda": loguniform(0.5, 2.0),
    "xgb__gamma": uniform(0.0, 2.0),
    "xgb__tree_method": ["hist", "approx"]
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
    halv_rand_xgb = HalvingRandomSearchCV(
        pipe,
        hyperparams,
        cv = k_,
        n_jobs = -1,
        verbose = 4,
        min_resources = 4 * k_ # Safe min row count to prevent the inner CV getting < 2 sample for validation
    )
    halv_rand_xgb.fit(x_train, y_train)

    # Predict and evaluate
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, halv_rand_xgb.predict(x_test_f)])