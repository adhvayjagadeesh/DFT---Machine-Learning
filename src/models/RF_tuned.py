from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.ensemble import RandomForestRegressor
from data.final import k_fold, k_
import numpy as np

# For combining predictions from all folds
y_pred = np.array([])
y_test = np.array([])

# Educated guess trust
hyperparams = {
    "n_estimators": Integer(100, 1000)
}

# By _f I mean fold
for x_train, y_train, x_test_f, y_test_f in k_fold():
    bayes_rf = BayesSearchCV(RandomForestRegressor(), hyperparams, cv = k_, n_jobs = -1, verbose = 4, n_iter = 10)
    bayes_rf.fit(x_train, y_train)

    # Predict and evaluate
    y_test = np.concatenate([y_test, y_test_f])
    y_pred = np.concatenate([y_pred, bayes_rf.predict(x_test_f)])