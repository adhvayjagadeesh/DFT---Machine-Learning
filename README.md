# DFT---Machine-Learning

Benchmarking ML models for predicting band gap of 2D material without DFT

## 1st time setup

```bash
# Make venv called ".venv"
python -m venv .venv
```

### Activate venv

Linux/WSL

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

### Finally

```bash
# Go to src directory
cd src

# Install dependencies
pip install -r requirements.txt
```

## Run + visualize 1 model

From `src`, run:

```bash
# Replace [name] with a filename in the models folder, without the .py
python -m stats.single [name]
```

## Model notes

- Define `y_pred` and `y_test` to support stats display.
- We are only reporting performance of model types, not creating a super good model, so use k-fold for base performance
- For XGB, using "dart" booster takes way too long, if u got time, try it
- Small compromise: To avoid the deadly triply-nested loop on hybrid models with weighting, we are going to use `split` to split 80% into hyperparam tuning or regular training, and 20% to weighting.
- When using `VotingRegressor` or `WeightedRegressor`, make sure to set the pipeline name to `""`, and name the models like the table below
- Four hybrid cases: K-fold (k_fold), weighting (w), bayesian (bayes), and both (both), although there's no both now
- 2-model hybrid matrix (empty = unimplemented, :no_entry_sign: = NOPE, :white_check_mark: = implemented, filename = \[row]_\[col]):

|      | RF                 | XGB                | GBT                | HGBT               | MLP                | SVR             |
|------|--------------------|--------------------|--------------------|--------------------|--------------------|-----------------|
| RF   | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| XGB  | :white_check_mark: | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| GBT  | :white_check_mark: | :white_check_mark: | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| HGBT | :white_check_mark: | :white_check_mark: | :white_check_mark: | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| MLP  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :no_entry_sign:    | :no_entry_sign: |
| SVR  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :no_entry_sign: |
