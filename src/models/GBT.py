from sklearn.ensemble import GradientBoostingRegressor
from data.final import train_test_split

x_train, y_train, x_test, y_test = train_test_split()

# Train GBT
gbt = GradientBoostingRegressor()
gbt.fit(x_train, y_train)

# Predict and evaluate
y_pred = gbt.predict(x_test)