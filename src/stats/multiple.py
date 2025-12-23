from argparse import ArgumentParser
from csv import writer as csv_writer
from os import getcwd, listdir, makedirs
from os.path import join
from re import compile
from shutil import rmtree

from stats.single import run

models = [
  file[:-3]
  for file in listdir(join(getcwd(), "models"))
  if file not in ("__pycache__", "__init__.py")
]
parser = ArgumentParser(
  "Run many models (defaults to all)",
  description="CSV with metrics and visual for multiple models",
)
parser.add_argument("res_dir", help="Result directory")

# Regex to include models to run
parser.add_argument("-s", "--select", help="Model selection regex")
args = parser.parse_args()

if args.select:
  ptn = compile(args.select)
  models = [i for i in models if ptn.fullmatch(i)]

res_dir = args.res_dir
rmtree(res_dir, True)
makedirs(res_dir)
model_cnt = len(models)
with open(join(res_dir, "result.csv"), "w", newline="") as res_csv:
  writer = csv_writer(res_csv)
  writer.writerow(
    ("Name", "R²", "Adj R²", "MAE (eV)", "RMSE (eV)", "Spearman", "Run time")
  )
  for i, model in enumerate(models, 1):
    print(f"Running {model} ({i}/{model_cnt})")
    try:
      writer.writerow(run(model, res_dir))
      res_csv.flush()
    except Exception as e:
      print(e)
