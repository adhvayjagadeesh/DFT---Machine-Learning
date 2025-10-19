from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.final import k_fold, k_
import numpy as np
from skopt.space import Real, Integer, Categorical

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor())
])

pipe_gbt = Pipeline([
    ("scaler", StandardScaler()),
    ("gbt", GradientBoostingRegressor())
])

# RandomForest hyperparameters (unchanged)
hyperparams_rf = {
    "rf__n_estimators": Integer(100, 1000),
    "rf__max_depth": Categorical([None, 10, 25, 50, 75, 100]),
    "rf__min_samples_split": Integer(2, 20),
    "rf__min_samples_leaf": Integer(1, 10),
    "rf__max_features": [1, "sqrt", "log2"],
    "rf__bootstrap": Categorical([True, False]),
}

# GradientBoosting hyperparameters (unchanged)
hyperparams_gbt = {
    "gbt__n_estimators": Integer(200, 800),
    "gbt__learning_rate": Real(1e-3, 0.2, prior="log-uniform"),
    "gbt__max_depth": Integer(3, 10),
    "gbt__min_samples_split": Integer(2, 15),
    "gbt__min_samples_leaf": Integer(1, 10),
    "gbt__subsample": Real(0.7, 1),
    "gbt__max_features": Categorical(["sqrt", 0.7, None]),
}

for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    # Tune RF (same as your RF_bayes.py)
    rand_rf = BayesSearchCV(
        pipe_rf,
        hyperparams_rf,
        cv = k_,
        n_iter = 20,
        n_jobs = -1,
        verbose = 4
    )
    rand_rf.fit(x_train, y_train)

    # Tune GBT (same as your GBT_bayes.py)
    bayes_gbt = BayesSearchCV(
        pipe_gbt,
        hyperparams_gbt,
        cv = k_,
        n_iter = 20,
        n_jobs = 1,
        verbose = 4
    )
    bayes_gbt.fit(x_train, y_train)

    # Predict and evaluate (simple average blend of the two models)
    y_test = np.concatenate([y_test, y_test_f])
    preds_rf = rand_rf.predict(x_test_f)
    preds_gbt = bayes_gbt.predict(x_test_f)
    preds_blend = (preds_rf + preds_gbt) / 2.0
    y_pred = np.concatenate([y_pred, preds_blend])
