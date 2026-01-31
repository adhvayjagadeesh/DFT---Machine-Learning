# DFT---Machine-Learning

ML models for predicting band gap of 2D material from elemental properties

## Setup

### Get the code

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

### Prepare virtual environment

```bash
# Make venv called ".venv" (with Python)
python -m venv .venv

# Make venv called ".venv" (with virtualenv)
virtualenv .venv

# Activate venv (Linux/WSL)
source .venv/bin/activate

# Activate venv (Windows)
.venv\Scripts\activate

# Install core packages and project in editable mode
pip install -e .
```

### Upgrade packages

```bash
# Install dev packages (1st time only)
pip install -e .[dev]

# Auto upgrade packages
pip-review --auto
```

### Run code asynchronously

```bash
# Create tmux session called "dftml" (1st time only)
tmux new-session -t dftml

# Attach that session (to check logs)
tmux a -t dftml

# Run code here
# Press "Ctrl-B" THEN "d" to detach session

# You can now turn of your local machine and disconnect
```

### Get files from ASDRP

Because ASDRP server setup is weird (stupid), where when user ssh to it:

- They first get connected to user@[master ip]
- Then a `.bash_profile` from `user@master` run `ssh user@[slave_ip]` to connect master to slave (on login)
- Hence the double-password, and it makes a transitive ssh connection (local --> master --> slave)

Let's define paths:

- src = source file on slave
- dst1 = destination on master
- dst2 = destination on local

```bash
# First, if you're in slave, get out
exit

# To get files from slave to local, first rsync it to the master. From master:
rsync -aP user@[slave ip]:[src] [dst1]

# Then from local, run:
rsync -aP user@[master ip]:[dst1] [dst2]
```

If you are working in master, then just do the last step

## Data aggregation

Aggregate a database into a parquet, and store them in the data folder. The first column is the target variable and the rest are features

### C2DB

```
usage: python -m data.aggregate_c2db [-h] db

Aggregate raw C2DB data

positional arguments:
  db          C2DB ASE database file

options:
  -h, --help  show this help message and exit
```

## Data exploration

### Feature distribution

Histogram and box plot for all numerical features

```
usage: python -m data.feat_dist [-h] [-s SAVE]

Show how numerical features are distributed

options:
  -h, --help       show this help message and exit
  -s, --save SAVE  Save the figure(s) instead of showing it
```

### Feature correlation

A heatmap of spearman correlation between numerical features

```
usage: python -m data.feat_correl [-h] [-s SAVE]

Show how numerical features are correlated

options:
  -h, --help       show this help message and exit
  -s, --save SAVE  Save the figure(s) instead of showing it
```

## Running model

<details>
<summary>Random state is fixed to seed 67 for reproducibility, put RANDOM=1 before run models to run with random_state = None</summary>

```
                    ⢀⣀⡤⠤⠖⠒⠒⠒⠒⠦⠖⠒⠒⠒⠒⠦⠤⣄⡀
                ⢀⣠⠴⠊⠉     ⣀           ⢈⠙⠦⣄
              ⢀⡴⠋  ⢠⢀⡄  ⢠ ⣇ ⢦  ⠸⡄     ⠈⠲⠄⠈⠓⢄
            ⢀⡜⠉⡀⢀⡴⣲⡏⣼⠁  ⡞ ⢹⡀⠘⣆  ⠹⡄       ⠙⢆⠈⠳⣄
            ⣰⠋⢀⢞⣴⢋⡜⣝⣼⠃  ⡼⠁ ⢸⠳⣄⠈⢷⣄⡀⠙⢦⡀      ⠈⠳⡀⠈⢣⡀
          ⡴⠃⢠⣾⣻⠗⣡⢾⡿⠁ ⢀⡼⠁ ⡰⠋ ⠈⠑⠦⢽⣻⣗⡦⢽⣦⣄⡀     ⠘⢦⡀⠹⡄
          ⢠⠇ ⣼⠋⠉ ⣩⠜⠁ ⣠⠞⢁⣠⠔⠃     ⠈⠉⠛⠃⠈⠉⠙⠓⠦⢄⡀   ⠈⠣⣀⠱⡄
        ⢀⡟ ⠐⠃⢀⡠⠞⠁⣀⡤⠞⠛⠋⠉        ⡀         ⠈⠓⢦   ⠈⠉⠹⣆⣆
        ⣼⠁  ⠙⠛⠛⡿⠉⠁⡤⢒⣯⣭⣭⣓⢤⡀  ⡇  ⢳  ⢀⡴⣺⣭⣯⣅⡓⢢⡀⠈⢧     ⠈⢣⡀
        ⡇     ⡼⠁ ⢞⢰⣿⣯⣿⣽⣿⠷⠃ ⢸⡁  ⢹⡀ ⠘⠿⣿⣏⣟⣺⣿⡆⢱ ⠈⢧     ⢀⣈⣓⡦⡄⡀
        ⡇    ⢰⠃   ⢀⣉⣉⣉⡡⠔⠃ ⣠⠞⠒⠉⠙⠂⠳⣄ ⠙⠦⢍⣉⣉⣉⡀   ⠘⣇     ⠙⠛⠒⢒⡿
  ⢠⡄    ⢰⠇   ⢀⡟         ⣠⡴⠋⢀⣬⣆ ⢰⣦⣀⡈⠱⡄⢄         ⢹⡄     ⢠⡶⠭⢾⠃⢀
  ⠈⣟⠶⢤⣀⡤⠟    ⢸       ⢀⡴⠊ ⠙⠤⠟⠋⠁ ⠈⠉⠛⠤⠤⠃ ⠙⠦⡀      ⠈⡇       ⢀⡞⢀⣡
⣴⡤⣈⣲⣤⡄      ⡏     ⢀⠔⠉                  ⠈⠲⡄     ⡇      ⢀⠞⠁    ⢳
⣿⣙⢄⡀       ⠠⡇    ⡰⠋   ⣀⣠⣤⠤⢤⣶⣶⠶⣶⣶⡤⠤⣤⢤⣀⣀   ⠈⢆    ⡇    ⣠⠔⣋⣠⠄   ⢀⣾
⠹⣌⠙⠛⠂      ⠨⡇   ⣸⠁ ⢠⣴⣿⠓⡞⠉⠱⠏  ⠿⠁ ⠈⠎⠉⢱⠚⢫⢽⡲⡀ ⠈⡆  ⢠⡇  ⣀⠈⢉⣩⠔⠋ ⢀⣰⣾⣿⣭
  ⣜⣣⣄⡀       ⣇  ⢰⡇ ⢰⡟⡇⡸ ⡄     ⢤   ⡀ ⣀ ⠸ ⡷⣽⡆ ⢳  ⢸ ⢀⣤⣙⣻⠿ ⢀⣶⣾⣿⣿⣿⣿⣿
  ⠈⠓⢯⣄⡀⡀     ⢹⡄  ⡇ ⣿⡹⡹⣜⣲⣿⣿⣿⣷⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⡭⠴⣠⢻⣷ ⢸⢀⡏⢀⡾⠉⠛⠲⣄⣾⣿⣿⣿⣿⠿⠟⠿⠿
    ⠈⠿⣗⡃     ⠘⣇  ⡇ ⣿⣧⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣯⣾⣿   ⣸⢃⡼⠃   ⠙⢯⡉⠉⠋⠉
      ⠈⠉⠛⠒⠒⠻⣦⡀⠹⡄ ⢳ ⣿⣿⣿⣿⣿⣿⣿  ⣿⣿⣿     ⣿⣿⣿⠐⠃⢀⡿⠋    ⢰⡀⠈⠻⢦⡁
          ⣠⡟⢹⡍⠓⢷   ⢻⣿⣿⣿⣿⣿⣿ ⣿⣿⣿⣿⣿⣿⣿⣿ ⣿⣿⣿⣿⡇  ⢸⠁      ⡇   ⢻⡀
        ⣠⠟⠁ ⠈⣇ ⢸⡄  ⢸⣿⣿⣿⣿⣿ ⣿⣿⣿⣿⣿⣿⣿⣿ ⣿⣿⣿⣿⣿⠇  ⡟      ⢰⠃   ⢸⣇
      ⢀⣤⣏    ⠹⡆ ⣇   ⣿⣿⣿⣿⣿ ⣿   ⣿⣿⣿⣿ ⣿⣿⣿⣿⣿⣿  ⢠⠇     ⢠⠏   ⢀⡾⠈⠛⢆
    ⢠⡶⠋⠁⠹⣆    ⠹⣄⢹   ⢻⣿⣿⣿⣿  ⣿⣿ ⣿⣿⣿ ⣿⣿⣿⣿⣿⣿⡟  ⣸    ⢀⡴⠋   ⣠⠟⠁  ⠈⢳⡀
  ⣴⠋    ⠈⢷⣄    ⠘⡇  ⢸ ⣿⣿⣿⣿⣿    ⣿⣿⣿ ⣿⣿⣿⣿⣿⣿⡇   ⡏    ⠈   ⢀⡼⠋      ⢳
  ⡼⠁       ⠙⢷⣄   ⢷  ⠘⡟⡛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣋⣹  ⢰⠇      ⢀⡴⠋
⣸⠁          ⠉⠳⣄⡀⢸⡀  ⣏⡀⢽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿ ⣸   ⢸     ⢀⡴⠋⠁
⢀⡇             ⠈⠉⠙⡇  ⢻⡀⠠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠤⣄⡇  ⡾   ⣠⠾⠋
⢸⠁                ⢿  ⠸⣦⠔⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⣀⢸⠇ ⢠⡇⣀⡴⠚⠁
⣸     ⠹⣇          ⢸⡄  ⢻⣀⠐⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢄ ⡿  ⢰⠟⠁
⣿      ⢿          ⠈⡇  ⠸⡏ ⡨⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡋⠘⣦⡇  ⣼
⣏      ⠘⣇          ⢷   ⠹⡜ ⡼⢿⣿⣿⣿⣿⣿⣿⠿⡏⠘⣄⡞   ⡏
⣧       ⢹⡄         ⢸⡀ ⠐⣄⠙⣶⡀⢸ ⢈⠉⡍ ⢡ ⣅⣠⠟⡀  ⢸⠇
⣷        ⢻⡄         ⣇  ⠈⢦⡈⠙⠫⠷⣶⣤⣖⣤⡾⠶⠋⢁⡴⠃ ⠠⡿
⢻         ⢻⡀        ⢹⡄   ⠈⠒⠤⣄⣀⣀⣀⣀⣀⡤⠔⠋   ⣸⠃
⢸          ⢻⣆        ⢳⣀     ⢀    ⡀⡀   ⢀⣰⠃
```

</details>

| Mode code | k      | t    | w      | wt              |
| --------- | ------ | ---- | ------ | --------------- |
| Meaning   | K-fold | Tune | Weight | Weight and tune |

| Model code | Meaning                                       |
| ---------- | --------------------------------------------- |
| rf         | Random forest                                 |
| xgb        | eXtreme Gradient Boosting                     |
| mlp        | Multi-Layer perceptron                        |
| gbt        | Gradient-boosted tree                         |
| hgbt       | Histogram-based GBT                           |
| adbt       | AdaBoost-ed decision tree                     |
| abet       | AdaBoost-ed extremely randomized (eXtRa) tree |
| ets        | eXtRa trees                                   |

Model name examples:

```bash
# Random forest with k-fold
rf_k

# Random forest and multi-layer perceptron weighted
rf+mlp_w
mlp+rf_w
```

### One model

An SVG with:

- Performace summary
- Predicted vs actual band gap scatterplot
- Error distribution histogram
- REC curve with AUC
- Learning curve
- Permutation feature importance bar graph with error bars.

| Feature # | Name                                           |
| --------- | ---------------------------------------------- |
| 1         | Energy above hull (eV/atom)                    |
| 2         | Heat of formation (eV/atom)                    |
| 3         | Magnetic                                       |
| 4         | Fermi level wrt. vacuum (PBE) (eV)             |
| 5         | Energy (eV)                                    |
| 6         | Magnetic anisotropy energy, xz (meV/unit cell) |
| 7         | Magnetic anisotropy energy, yz (meV/unit cell) |
| 8         | Vacuum level (eV)                              |

```
usage: python -m stats.one [-h] [-s SAVE] name

Visualization and stats for a model

positional arguments:
  name

options:
  -h, --help       show this help message and exit
  -s, --save SAVE  Save the figure instead of showing it
```

### Many models

```
usage: python -m stats.many [-h] res_dir names [names ...]

CSV with metrics (in a CSV) and visual for many models

positional arguments:
  res_dir     Result directory
  names

options:
  -h, --help  show this help message and exit
```

## Explanations

### Shuffling

See [this](https://stats.stackexchange.com/questions/629193/does-k-fold-cross-validation-strictly-require-shuffling-of-data-before-splitting). In summary, we must do `KFold(shuffle=True)`, but this makes it copy our data, so instead, we can shuffle the data file itself (with `shuf`) and then do `KFold()`.

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

### Learning curve

A learning curve plots the model's error (or performance) vs the training set size. It answers the question: "Will adding more data improve the model?" We use error in our case, so lower is better as opposed to using performance where higher is better

| Scenario | Training error (bias) | Testing error (variance) | Meaning: Model...                                                             |
| -------- | --------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| Overfit  | Very Low              | High                     | Captured noise instead of physics (too complex).                              |
| Underfit | High                  | High                     | Can't capture the underlying pattern (too simple)                             |
| Good Fit | Low (acceptable)      | Low                      | Captured the underlying pattern while being robust to noise (good complexity) |

### Feature importance

There are two kinds of feature importance that we can use: impurity-based (MDI) and permutation. MDI importance calculates importance based on how much a feature reduces impurity (like Gini impurity or entropy) during tree building. Some problems with MDI include:

- Only for tree-based models
- Based on training-data because it was constructed during tree building
- Biased (makes a feature more important than it actually is) towards high-cardinality (many unique values) feature because trees naturally split on more unique values to reduce impurity

Permutation importance solves all of these problem by measuring change in error on the test set when one feature (its rows in the dataset) is shuffled. How much error increase will tell how important that feature is. If error decreases or barely increases, that feature should be eliminated.

## Developer notes

- Any file that depends on a random process must import `seeder` to ensure reproducibility.
- To generate correct help messages, program names for Argparse should follow `python -m [module name]` such as `python -m data.feat_correlation`
- `VotingRegressor`'s weights can be reset with `set_params` without refitting (see test.py)
- We are only reporting performance of model types, not creating a super good model, so use k-fold for base performance
- Takes way too long (removed):
  - XGB with `dart` booster
  - SVR
- XGB doesn't like that feature (column) names has `[`, `]` or `<` so I renamed the columns in the CSV file, square bracket to parenthesis
- Our model are always assumed to be a `Pipeline` with a first step of `("scaler", RobustScaler())` and the second of a `("", VotingRegressor)` whether it's an ensemble or not
