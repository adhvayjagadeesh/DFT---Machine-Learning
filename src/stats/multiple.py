from argparse import ArgumentParser
from csv import writer as csv_writer
from os import makedirs
from os.path import join
from shutil import rmtree

from stats.one import possible_modes, possible_names, run_visualize


def _all_possible_models() -> list[tuple[str, list[str]]]:
  models = []

  # Single-model
  for mode in [mode for mode in possible_modes if "w" not in mode]:
    for name in possible_names:
      models.append((mode, [name]))

  return models


parser = ArgumentParser(
  "python -m stats.multiple",
  description="CSV with metrics (in a CSV) and visual for multiple models",
)
parser.add_argument("res_dir", help="Result directory")
args = parser.parse_args()

res_dir = args.res_dir
rmtree(res_dir, True)
makedirs(res_dir)
with open(join(res_dir, "result.csv"), "w", newline="") as res_csv:
  writer = csv_writer(res_csv)
  writer.writerow(
    ("Name", "R²", "Adj R²", "MAE (eV)", "RMSE (eV)", "Spearman", "Fit time")
  )
  models = _all_possible_models()
  n_model = len(models)
  for i, (mode, names) in enumerate(models, 1):
    print(f"Running {'+'.join(names)}_{mode} ({i}/{n_model})")
    try:
      writer.writerow(run_visualize(mode, names, res_dir))
      res_csv.flush()
    except Exception as e:
      print(e)
