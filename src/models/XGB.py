from xgboost import XGBRegressor
from data.final import train_test_split

x_train, y_train, x_test, y_test = train_test_split()

# Train RF
xgb = XGBRegressor()
xgb.fit(x_train, y_train)

# Predict and evaluate
y_pred = xgb.predict(x_test)