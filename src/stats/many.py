from argparse import ArgumentParser
from os import makedirs

from stats.one import metric_names, run_visualize
from stats.save import save_tbl

parser = ArgumentParser(
  "python -m stats.many",
  description="Metric and visual for many models",
)
parser.add_argument("output", help="Output directory")
parser.add_argument("names", nargs="+", help="Model names to run")
args = parser.parse_args()

output = args.output
makedirs(output, exist_ok=True)

names = args.names
n_to_run = len(names)

metrics = []

for i, name in enumerate(names):
  print(f"Running {name} ({i + 1}/{n_to_run})")
  try:
    metrics.append(run_visualize(name, output))
  except Exception as e:
    print(e)
save_tbl(output, "metrics", metrics, metric_names)
