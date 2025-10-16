from sklearn.ensemble import RandomForestRegressor
from data.final import train_test_split

x_train, y_train, x_test, y_test = train_test_split()

# Train RF
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

# Predict and evaluate
y_pred = rf.predict(x_test)