from argparse import ArgumentParser
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from data.load import x, y

parser = ArgumentParser(
  "python -m data.feat_correlation",
  description="Show how numerical features are correlated",
)
parser.add_argument(
  "-s", "--save", help="Save the figure(s) instead of showing it"
)
args = parser.parse_args()

# All numerical features + target
x.drop(columns=["Magnetic"], inplace=True)
data = chain(x.items(), y.to_frame().items())


# 2. Create a mask for the upper triangle
# np.triu generates a matrix with ones in the upper triangle and zeros elsewhere
mask = np.triu(np.ones_like(data, dtype=bool))

# 3. Apply the mask to the data (set masked values to NaN or some other indicator)
# Matplotlib's imshow handles masked arrays by leaving masked areas white/transparent
masked_data = np.ma.masked_where(mask, data)

# 4. Plot using imshow
plt.imshow(masked_data, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Value")
plt.title("Half Heatmap with Matplotlib imshow")
plt.show()
