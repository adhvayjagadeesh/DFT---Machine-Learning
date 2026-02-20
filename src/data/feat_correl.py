from argparse import ArgumentParser

import matplotlib.pyplot as plt
from numpy import array, tril

from data.load import df
from stats.save import save_fig

parser = ArgumentParser(
  "python -m data.feat_correl",
  description="Show how numerical features are correlated",
)
parser.add_argument("-o", "--output", help="Folder to save output instead of showing it")
args = parser.parse_args()
correls = df.corr("spearman")
vars = array(correls.columns)
correls = tril(correls.values[1:, :-1])
var_range = range(len(correls))


fig, ax = plt.subplots()

# Add correlation strength for nonzero correlations
for i in var_range:
  for j in var_range:
    correl = correls[i][j]
    if not correl == 0:
      ax.text(
        j,
        i,
        f"{correl:.4f}",
        ha="center",
        va="center",
        # Contrast
        c="#000000" if abs(correl) < 0.25 else "#ffffff",
      )

# Choose a colormap that's white for 0
im = ax.imshow(correls, cmap="seismic", vmin=-1, vmax=1)

# Color bar to indicate correlation strength
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Spearman correlation", rotation=-90, va="bottom")

ax.set_title("Feature correlation heatmap")

# Rotate to save vertical space and make reading easier
ax.set_xticks(var_range, vars[:-1], rotation=30, rotation_mode="anchor", ha="right")

ax.set_yticks(var_range, vars[1:])
ax.spines[:].set_visible(False)

plt.tight_layout()
save_fig(args.output, "feat_correl", plt)
