# DFT---Machine-Learning

Benchmarking ML models for predicting band gap of 2D material without CBM & VBM

## Get the code

- Cloning with SSH is the easiest here. [Setup guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

```bash
git clone git@github.com:ASDRP-DFT-Machine-Learning/DFT---Machine-Learning.git dftml
```

- After cloning or if you already have the repo

```bash
# Go to the repo
cd dftml

# Update (if you didn't just clone)
git pull
```

## Prepare virtual environment

```bash
# Make venv called ".venv" (with Python)
python -m venv .venv

# Make venv called ".venv" (with virtualenv)
virtualenv .venv

# Activate venv (Linux/WSL)
source .venv/bin/activate

# Activate venv (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Run code asynchronously

```bash
# Create tmux session called "dftml" (1st time only)
tmux new-session -t dftml

# Attach that session (to check logs)
tmux a -t dftml

# Run code here
# Press "Ctrl-B" THEN "d" to detach session

# You can now turn of your local machine and disconnect
```

## Get files from ASDRP

Because ASDRP server setup is weird (stupid), where when user ssh to it:

- They first get connected to user@[master ip]
- Then a bash profile from user@master run `ssh user@[slave_ip]` to connect master to slave (on login)
- Hence the double-password, and it makes a transitive ssh connection (local --> master --> slave)

Let's define paths:

- src = source file on slave
- dst1 = destination on master
- dst2 = destination on local

```bash
# First, if you're in slave, get out
exit

# To get files from slave to local, first rsync it to the master. From master:
rsync -avP user@[slave ip]:[src] [dst1]

# Then from local, run:
rsync -avP user@[master ip]:[dst1] [dst2]
```

If you are working in master, then just do the last step

## Run 1 model

```bash
# Replace [name] with a filename in the models folder, without the .py
python -m stats.single [name]
```

## Run multiple models

```bash
# Replace [result_dir] with output directory relative to src
# It will run all models, use -s/--select to use regex to choose what to run
python -m stats.multiple [result_dir]
```

## Developer notes

- Define `y_pred` and `y_test` to support stats display.
- We are only reporting performance of model types, not creating a super good model, so use k-fold for base performance
- For XGB, using "dart" booster takes way too long, if u got time, try it
- Small compromise: To avoid the deadly triply-nested loop on hybrid models with weighting, we are going to use `split` to split 80% into hyperparam tuning or regular training, and 20% to weighting
- When using `VotingRegressor` make sure to set the pipeline name to `""`, and name the models like the table below
- 6 model types:
  - 2 Standalone: K-fold (k) and bayesian-optimized (b)
  - 4 Hybrid: K-fold, weighting (w), bayesian-optimized, and weighted + bayesian (wb)
- To avoid repeating comments, only XGB\_\* files are fully commented (because it has exactly these 6 types)
- 2-model hybrid matrix (empty = unimplemented, :no*entry_sign: = NOPE, :white_check_mark: = implemented, filename = \[rows]*\[col]):

|      | RF                 | XGB                | GBT                | HGBT               | MLP                | SVR             |
| ---- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | --------------- |
| RF   | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| XGB  | :white_check_mark: | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| GBT  | :white_check_mark: | :white_check_mark: | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| HGBT | :white_check_mark: | :white_check_mark: | :white_check_mark: | :no_entry_sign:    | :no_entry_sign:    | :no_entry_sign: |
| MLP  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :no_entry_sign:    | :no_entry_sign: |
| SVR  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :no_entry_sign: |
