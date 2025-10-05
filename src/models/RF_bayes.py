# RF tuned with Bayesian Optimization (warning: can be painfully slow)

from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from data.final import k_fold, k_
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# Educated guesses (from previous runs)
hyperparams = {
    "n_estimators": [100, 600, 700, 1000],
    "max_depth": [None, 25, 50, 75, 100]
}

# By _f I mean fold
for x_train, y_train, x_test_f, y_test_f in k_fold():
    bayes_rf = BayesSearchCV(RandomForestRegressor(),
                            hyperparams, 
                            cv = k_,
                            n_jobs = -1,
                            n_iter = 20,
                            verbose = 0)
    bayes_rf.fit(x_train, y_train)
    print(f"Fold's best params: {bayes_rf.best_params_}")

    
    # Predict and evaluate
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, bayes_rf.predict(x_test_f)])