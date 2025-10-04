# This is special because LassoCV will automagically find the optimal alpha, so
# k-fold done internally already, we can just use train-test-split

from sklearn.linear_model import LassoCV
from data.final import train_test_split, rand_seed, k_

x_train, y_train, x_test, y_test = train_test_split()

# Train LASSO
lasso = LassoCV(cv=k_)
lasso.fit(x_train, y_train)

# Predict and evaluate
y_pred = lasso.predict(x_test)