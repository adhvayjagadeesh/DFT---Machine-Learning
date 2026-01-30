from os import environ

from numpy.random import seed

if "RANDOM" not in environ:
  # Fixed random seed for reproducibility
  seed(67)  # SIX SEVEN

# Import this file where you need to fix a seed
