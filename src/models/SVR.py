from sklearn.svm import SVR
from data.final import train_test_split

x_train, y_train, x_test, y_test = train_test_split()

# Train SVR
svr = SVR()
svr.fit(x_train, y_train)

# Predict and evaluate
y_pred = svr.predict(x_test)