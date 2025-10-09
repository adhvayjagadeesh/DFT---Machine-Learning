from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from data.final import k_fold, k_
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform, randint
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(max_iter = 1000000)) # Some of the cv is taking too long
])

hyperparams = {
    'svr__C': loguniform(1e-3, 1e+6),
    'svr__gamma': loguniform(1e-6, 1e+1),
    "svr__degree": randint(1, 9),
    "svr__epsilon": loguniform(1e-4, 1e-1),
    "svr__kernel": ["linear", "poly", "rbf"]
}

# By _f I mean fold, no scaling cuz pipeline is gonna handle that
for x_train, y_train, x_test_f, y_test_f in k_fold(scale = False):
    rand_svr = RandomizedSearchCV(pipe,
                            hyperparams, 
                            cv = k_,
                            n_jobs = -1,
                            n_iter = 20,
                            verbose = 4)
    rand_svr.fit(x_train, y_train)

    # Predict and evaluate
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, rand_svr.predict(x_test_f)])