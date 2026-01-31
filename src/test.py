from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor

b = 0


# Test estimator
class Testimator(RegressorMixin, BaseEstimator):
  def fit(self, x, y):
    pass

  def predict(self, x):
    global b
    b += 2
    return [b]


a = VotingRegressor([("1", Testimator()), ("2", Testimator())])
a.fit([1], [1])

# .5 * 2 + .5 * 4 = 3
assert a.predict([1])[0] == 3

# Reset for next test
b = 0

# Set weights without refitting
a.set_params(weights=[0, 1])

# 0 * 2 + 1 * 4 = 4
assert a.predict([1])[0] == 4
