# DFT---Machine-Learning

## 1st time setup
```bash
# Make venv called ".venv"
python -m venv .venv

# Activate the venv
source .venv/bin/activate

# Go to src directory
cd src

# Install deps
pip install -r requirements.txt
``` 

## Run + visualize 1 model (WIP)
From `src`, run:
```bash
# Replace [name] with a filename in the models folder, without the .py
python -m visualization.single [name]
```