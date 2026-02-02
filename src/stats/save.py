from matplotlib import get_backend
from tabulate import tabulate

backend = get_backend()


# Save or show figure
def save_fig(output, name, fig):
  if output:
    fig.savefig(f"{output}/{name}.pgf", bbox_inches="tight")
  elif backend != "agg":
    fig.show()
  else:
    # Failsafe for "UserWarning: FigureCanvasAgg is non-interactive, and thus cannot
    # be shown" to prevent forgetting graphics backend and losing 67 hours of progress.
    print(
      "IMMEDIATELY install and set the Matplotlib interactive backend .\n",
      "I hereby rescue you this time and this time only. INSTALL IT NOW.\n",
      f"Figured saved as {name}.pgf",
    )
    fig.savefig(f"./{name}.pgf", bbox_inches="tight")


# Save or print table
def save_tbl(output, name, data, headers):
  tbl = tabulate(data, headers=headers, tablefmt="latex" if output else "simple")
  if output:
    with open(f"{output}/{name}.tex", "w") as f:
      f.write(tbl)
  else:
    print(tbl)
