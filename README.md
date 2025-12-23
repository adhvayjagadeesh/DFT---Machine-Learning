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

Put `RANDOM=67` before running 1 or more models to run with `random_state = None`

```bash
# Replace [name] with a filename in the models folder, without the .py
python -m stats.single [name]
```

## Run multiple models

Put `RANDOM=67` before running 1 or more models to run with `random_state = None`

```bash
# Replace [result_dir] with output directory relative to src
# It will run all models, use -s/--select to use regex to choose what to run
python -m stats.multiple [result_dir]
```

## Developer notes

- `VotingRegressor`'s weights can be reset with `set_params` without refitting (see test.py)
- We are only reporting performance of model types, not creating a super good model, so use k-fold for base performance
- For XGB, using "dart" booster takes way too long, if u got time, try it
- Our model are always assumed to be a `Pipeline` with a first step of `("scaler", RobustScaler())` and the second of a `("", VotingRegressor)`
- 2 Standalone: K-fold and tuned (t)
- 4 Hybrid: K-fold, weighting (w), tuned, and weight + tune (wt)

## Explanations

### Tuning

Joint-tuning is when we:

- Make an ensemble with all our models
- Grab the hyperparameter search space for it
- Tune the whole ensemble with those hyperparameters

Individual-tuning is when we:

- For each model, we:
  - Grab the hyperparameter search space
  - Tune the model with those hyperparameters
- Put all the tuned models into an ensemble

Theoretically, joint-tuning is better than inidividual-tuning because the "team chemistry" argument where one ensemble can cover the weakness of another. In this case, imagine one consistently underestimate, while another consistent overestimates such that an appropriate weight will cancel them out and give a good estimate. Individual-tuning is like creating an "all-star" team that can make the same mistake (on difficult/edge cases).

Practically, joint-tuning fails because of dimensionality. If we have 4 models with 5 hyperparameters each, individual tuning requires solving four separate 5-dimensional optimization problems (manageable). Joint-tuning combines them into a single 20-dimensional problem. The probability of tuning and finding a decent dip in the loss function (input: n-dimension-hyperparameters, output: squared error) is very low, so very likely, we would land somewhere mediocre instead of that perfect cancellation dip.

Individual-tuning, does not increase the dimensionality of the search space so tuning will be more likely to find a very good dip in loss function for each model (because of the reduced search space), which correlate to maybe a decent dip in the ensemble loss function when putting them toghether. Individual-tuning usually yields a better real-world ensemble.

## Optimizing weights

We optimize weights by having
