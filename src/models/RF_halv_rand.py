# idk y, but this bro is not deterministic, maybe it's cuz it's experimental

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_, rng_seed
import numpy as np
from scipy.stats import randint

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor())
])

hyperparams = {
    "rf__n_estimators": randint(100, 1001),
    "rf__max_depth": [None, 10, 25, 50, 75, 100],
    "rf__min_samples_split": randint(2, 21),
    "rf__min_samples_leaf": randint(1, 11),
    "rf__max_features": [1, "sqrt", "log2"],
    "rf__bootstrap": [True, False],
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale=False):
    halv_rand_rf = HalvingRandomSearchCV(
        pipe,
        hyperparams,
        cv = k_,
        n_jobs = -1,
        verbose = 4,
        min_resources = 4 * k_ # Safe min row count to prevent the inner CV getting < 2 sample for validation
    )
    halv_rand_rf.fit(x_train, y_train)

    # Predict and evaluate
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, halv_rand_rf.predict(x_test_f)])