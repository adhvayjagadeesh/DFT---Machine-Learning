from matplotlib import get_backend

backend = get_backend()


def save_fig(save_loc, name, fig):
  if save_loc:
    fig.savefig(f"{save_loc}/{name}.pgf", bbox_inches="tight")
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
