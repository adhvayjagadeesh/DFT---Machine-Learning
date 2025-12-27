from argparse import ArgumentParser
from csv import writer as csv_writer
from os import makedirs
from os.path import join
from shutil import rmtree

from stats.one import run_visualize

parser = ArgumentParser(
  "python -m stats.many",
  description="CSV with metrics (in a CSV) and visual for many models",
)
parser.add_argument("res_dir", help="Result directory")
parser.add_argument("names", nargs="+")
args = parser.parse_args()

res_dir = args.res_dir
rmtree(res_dir, True)
makedirs(res_dir)
with open(join(res_dir, "result.csv"), "w", newline="") as res_csv:
  writer = csv_writer(res_csv)
  writer.writerow(
    (
      "Name",
      "R²",
      "Adjusted R²",
      "MAE (eV)",
      "RMSE (eV)",
      "Spearman",
      "Fit time",
    )
  )
  n_to_run = len(args.names)
  for i, name in enumerate(args.names, 1):
    print(f"Running {name} ({i}/{n_to_run})")
    try:
      writer.writerow(run_visualize(name, res_dir))
      res_csv.flush()
    except Exception as e:
      print(e)
