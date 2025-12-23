from enum import Enum

from sklearn.ensemble import (
  GradientBoostingRegressor,
  HistGradientBoostingRegressor,
  RandomForestRegressor,
  VotingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


class Model(Enum):
  rf = RandomForestRegressor
  xgb = XGBRegressor
  mlp = MLPRegressor
  svr = SVR
  gbt = GradientBoostingRegressor
  hgbt = HistGradientBoostingRegressor


def create_model(names: tuple[str, ...]) -> Pipeline:
  models = [(i, (Model[i].value)()) for i in names]
  return Pipeline(
    [
      ("scaler", RobustScaler()),
      ("", VotingRegressor(models)),
    ]
  )
