# DFT---Machine-Learning

## 1st time setup
```bash
# Make venv called ".venv"
python -m venv .venv

# Activate the venv
source .venv/bin/activate

# Go to src directory
cd src

# Install dependencies
pip install -r requirements.txt
```

## Run + visualize 1 model (WIP)
From `src`, run:
```bash
# Replace [name] with a filename in the models folder, without the .py
python -m visualization.single [name]
```

# Model notes
- Define `y_pred` and `y_test` to support visualization
- We are only reporting performance of model types, not creating a super good model, so prefer k-fold