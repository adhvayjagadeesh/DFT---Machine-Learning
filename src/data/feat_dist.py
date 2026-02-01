from argparse import ArgumentParser

import matplotlib.pyplot as plt

from data.load import df
from stats.save_fig import save_fig

parser = ArgumentParser(
  "python -m data.feat_dist",
  description="Show how numerical features are distributed",
)
parser.add_argument(
  "-s", "--save", help="Save the figure(s) instead of showing it"
)
args = parser.parse_args()

# Plotting
rows = 2
cols = 2
bins = 32
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
box_height = 0.175


def save_or_show(page):
  plt.tight_layout()
  save_fig(args.save, f"feat_dist_{page}", plt)
  global fig, axes
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4))


i = 0
for i, (name, vals) in enumerate(df.items()):
  if i > 0 and i % n == 0:
    save_or_show(i // n)

  row = (i % n) // cols
  col = (i % n) % cols
  ax = axes[row][col]

  # Histogram
  freq, _, _ = ax.hist(
    vals, bins=bins, alpha=0.7, color="teal", edgecolor="black"
  )

  # Add vertical space above the highest bar for boxplot
  ax.set_ylim(0, max(freq) * (1.05 + box_height))

  # Boxplot using inset axes
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
