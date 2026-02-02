from argparse import ArgumentParser
from csv import writer as csv_writer
from os import makedirs, scandir
from sys import exit

from stats.one import metric_names, run_visualize

parser = ArgumentParser(
  "python -m stats.many",
  description="CSV with metrics and visual for many models",
)
parser.add_argument("output", help="Output directory")
parser.add_argument("names", nargs="+", help="Model names to run")
args = parser.parse_args()
output = args.output
makedirs(output)
empty = True
for _ in scandir(output):
  empty = False
  break
if not empty:
  print("Output folder is not empty")
  exit(1)

with open(join(output, "result.csv"), "w", newline="") as res_csv:
  writer = csv_writer(res_csv)
  writer.writerow(metric_names)
  n_to_run = len(args.names)
  for i, name in enumerate(args.names, 1):
    print(f"Running {name} ({i}/{n_to_run})")
    try:
      writer.writerow(run_visualize(name, output))
      res_csv.flush()
    except Exception as e:
      print(e)
