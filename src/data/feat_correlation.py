from argparse import ArgumentParser
from itertools import chain

import matplotlib.pyplot as plt

from data.prepare import x, y

parser = ArgumentParser(
  "python -m data.feat_correlation",
  description="Show how numerical features are correlated",
)
parser.add_argument(
  "-s", "--save", help="Save the figure(s) instead of showing it"
)
args = parser.parse_args()

# All numerical features + HSE06 band gap
data = chain(x.drop(columns=["Magnetic"]).items(), y.to_frame().items())
