# Make box plot + histogram for each
import itertools
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from data.final import x, y

parser = ArgumentParser(
  "Feature distribution histograms", description="Show how features are distributed"
)
parser.add_argument("-s", "--save", help="Save the figure(s) instead of showing it")
args = parser.parse_args()

# All features + HSE06 band gap
data = itertools.chain(
  x.drop(columns=["Formula", "Magnetic"]).items(), y.to_frame().items()
)

# Plotting
rows = 2
cols = 2
bins = 32
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4))
box_height = 0.175


def save_or_show(page):
  plt.tight_layout()
  if args.save:
    plt.savefig(f"{args.save}/{page}.svg")
  else:
    plt.show()
  global fig, axes
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4))


for i, (name, vals) in enumerate(data):
  if i > 0 and i % n == 0:
    save_or_show(i // n)

  row = (i % n) // cols
  col = (i % n) % cols
  ax = axes[row][col]

  # Histogram
  counts, _, _ = ax.hist(vals, bins=bins, alpha=0.7, color="teal", edgecolor="black")

  # Add vertical space above the highest bar
  ymax = max(counts)
  ax.set_ylim(0, ymax * (1.05 + box_height))

  # Boxplot above histogram using inset axes
  ax_box = ax.inset_axes([0, 1 - box_height, 1, box_height])
  ax_box.boxplot(
    vals,
    vert=False,
    widths=0.75,
    patch_artist=True,
    boxprops=dict(facecolor="orange", alpha=0.7),
  )
  ax_box.set_xticks([])
  ax_box.set_yticks([])
  ax_box.set_frame_on(False)

  ax.set_title(name)

# Show/save the last page
save_or_show(i // n + 1)
