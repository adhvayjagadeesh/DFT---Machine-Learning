from argparse import ArgumentParser

import matplotlib.pyplot as plt
from numpy import tril

from data.load import df

parser = ArgumentParser(
  "python -m data.feat_correl",
  description="Show how numerical features are correlated",
)
parser.add_argument(
  "-s", "--save", help="Save the figure(s) instead of showing it"
)
args = parser.parse_args()
correls = df.corr("spearman")
var_names = list(correls.columns)
correls = tril(correls.values, k=-1)
var_range = range(len(correls))


fig, ax = plt.subplots()

# Add correlation strength for nonzero correlations
for i in var_range:
  for j in var_range:
    correl = correls[i][j]
    if not correl == 0:
      ax.text(j, i, f"{correl:.4f}", ha="center", va="center")

# Choose a colormap that's white for 0
im = ax.imshow(correls, cmap="bwr", vmin=-1, vmax=1)

# Color bar to indicate correlation strength
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Spearman correlation", rotation=-90, va="bottom")

ax.set_title("Feature correlation heatmap")

# Rotate to save vertical space and make reading easier
ax.set_xticks(
  var_range, var_names, rotation=45, rotation_mode="anchor", ha="right"
)

ax.set_yticks(var_range, var_names)
ax.spines[:].set_visible(False)

plt.tight_layout()
plt.show()
