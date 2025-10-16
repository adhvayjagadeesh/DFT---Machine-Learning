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

# Model notes
- Define `y_pred` and `y_test` to support stats display
- We are only reporting performance of model types, not creating a super good model, so use k-fold for base performance
- For XGB, using "dart" booster takes way too long, if u got time, try it
