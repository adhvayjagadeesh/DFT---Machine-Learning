# DFT---Machine-Learning

Benchmarking ML models for predicting band gap of 2D material without DFT

## Local setup

```bash
# Make venv called ".venv"
python -m venv .venv

# Activate venv (Linux/WSL)
source .venv/bin/activate
`
# Activate venv (Windows)
.venv\Scripts\activate

# Go to src directory
cd src

# Install dependencies (1st time only)
pip install -r requirements.txt
```

## ASDRP remote setup

Assuming you ssh to ASDRP server already

```bash
# Clone the repo (1st time only)
git clone https://github.com/adhvayjagadeesh/DFT---Machine-Learning dftml

# Go to the repo
cd dftml

# Update it
git pull

# Make venv called ".venv" (1st time only)
python -m venv .venv

# Activate venv
source .venv/bin/activate

# Go to src directory
cd src

# Install dependencies (1st time only)
python -m pip install -r requirements.txt
```

Run code asynchronously

```bash
# Create tmux session called "dftml" (1st time only)
tmux new-session -t dftml

# Activate venv (inside the tmux this time)
source .venv/bin/activate

# Install dependencies (1st time only)
python -m pip install -r requirements.txt

# Attach that session (to check logs)
tmux a -t dftml

# Run code here
# Press "Ctrl-B" THEN "d" to detach session

# You can now turn of your local machine and disconnect
```

## Get files from ASDRP

Because ASDRP server setup is weird (stupid), where when user ssh to it:

- They first get connected to user@[master ip]
- Then a bash profile from user@master run `ssh user@[slave_ip]` to connect master to slave
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

## Run 1 model

```bash
# Replace [name] with a filename in the models folder, without the .py
python -m stats.single [name]
```

## Run multiple models

```bash
# Replace [result_dir] with output directory relative to src
# It will run all models, add -e/--exclude [models] to skip them
python -m stats.multiple [result_dir]
```

## Developer notes

- A Python venv requires Python version >3.5, but ASDRP only have <3.12, so all code must be written for 3.5 < Python version < 3.12
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
