from sklearn.ensemble import HistGradientBoostingRegressor
from data.final import train_test_split

x_train, y_train, x_test, y_test = train_test_split()

# Train HGBT
hgbt = HistGradientBoostingRegressor()
hgbt.fit(x_train, y_train)

# Predict and evaluate
y_pred = hgbt.predict(x_test)