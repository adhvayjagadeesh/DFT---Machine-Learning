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
models = [file[:-3] for file in listdir(join(getcwd(), "models")) 
  if file not in ("__pycache__", "__init__.py")]
model_cnt = len(models)
with open(join(res_dir, "result.csv"), 'w', newline = '') as res_csv:
  writer = csv_writer(res_csv)
  writer.writerow(("Name", "R^2", "Adj R^2", "MAE (eV)", "RMSE (eV)", "Spearman"))
  for i, model in enumerate(models, 1):
    print(f"Running {model} ({i}/{model_cnt}")
    writer.writerow(run_model(model, res_dir))
    res_csv.flush()