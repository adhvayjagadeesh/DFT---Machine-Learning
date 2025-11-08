from os import listdir, getcwd
from os.path import join
from single import run_model
from argparse import ArgumentParser

parser = ArgumentParser("Run all models", description = "CSV with metrics and visual for all models")
parser.add_argument("result_dir")
res_dir = parser.parse_args().result_dir
models = [file[:-3] for file in listdir(join(getcwd(), "models"))]
for model in models:
  run_model(model, res_dir)