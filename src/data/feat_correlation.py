import itertools
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from data.prepare import x, y

parser = ArgumentParser(
  "python -m data.feat_correl",
  description="Show how features are distributed",
)
parser.add_argument(
  "-s", "--save", help="Save the figure(s) instead of showing it"
)
args = parser.parse_args()

# All numerical features + HSE06 band gap
data = itertools.chain(
  x.drop(columns=["Magnetic"]).items(), y.to_frame().items()
)
