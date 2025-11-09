from os import listdir, getcwd, makedirs
from os.path import join
from shutil import rmtree
from stats.single import run_model
from argparse import ArgumentParser
from csv import writer as csv_writer

parser = ArgumentParser("Run all models", description = "CSV with metrics and visual for all models")
parser.add_argument("result_dir")
res_dir = parser.parse_args().result_dir
rmtree(res_dir, True)
makedirs(res_dir)
models = [file[:-3] for file in listdir(join(getcwd(), "models"))]
with open(join(res_dir, "result.csv"), 'w', newline = '') as res_csv:
  writer = csv_writer(res_csv)
  writer.writerow(("Name", "R^2", "Adj R^2", "MAE", "RMSE", "Spearman"))
  for model in models:
    print(f"Running {model}")
    writer.writerow(run_model(model, res_dir))
    res_csv.flush()