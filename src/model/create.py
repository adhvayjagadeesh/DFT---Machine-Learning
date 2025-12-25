from enum import Enum
from os import cpu_count

from sklearn.ensemble import (
  GradientBoostingRegressor,
  HistGradientBoostingRegressor,
  RandomForestRegressor,
  VotingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


class Model(Enum):
  rf = RandomForestRegressor
  xgb = XGBRegressor
  mlp = MLPRegressor
  gbt = GradientBoostingRegressor
  hgbt = HistGradientBoostingRegressor


n_jobs = cpu_count()


def create_model(names: tuple[str, ...]) -> Pipeline:
  models = [(name, Model[name].value(n_jobs=n_jobs)) for name in names]
  return Pipeline(
    [
      ("scaler", RobustScaler()),
      ("", VotingRegressor(models)),
    ]
  )
