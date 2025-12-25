from enum import Enum

from sklearn import clone
from sklearn.ensemble import (
  AdaBoostRegressor,
  ExtraTreesRegressor,
  GradientBoostingRegressor,
  HistGradientBoostingRegressor,
  RandomForestRegressor,
  VotingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor


class Model(Enum):
  rf = RandomForestRegressor()
  xgb = XGBRegressor()
  mlp = MLPRegressor()
  gbt = GradientBoostingRegressor()
  hgbt = HistGradientBoostingRegressor()
  abdt = AdaBoostRegressor(DecisionTreeRegressor())
  abet = AdaBoostRegressor(ExtraTreeRegressor())
  ets = ExtraTreesRegressor()


possible_modes = ["k", "w", "t", "wt"]
possible_names = list(Model.__members__)


def parse_name(name):
  joined_names, mode = name.split("_", 1)
  names = joined_names.split("+")
  assert mode in possible_modes
  for name in names:
    assert name in possible_names

  # Weighting is only for ensemble
  if len(names) < 2:
    assert "w" not in mode

  return names, mode


def create_model(names: tuple[str, ...]) -> Pipeline:
  models = [(name, clone(Model[name].value)) for name in names]
  return Pipeline(
    [
      ("scaler", RobustScaler()),
      ("", VotingRegressor(models)),
    ]
  )
