import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load and clean data
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")

# Drop high-null and specified columns
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=[
  'Direct band gap (PBE) [eV]', 'Direct band gap (PBE) [eV].1',
  'Band gap (PBE) [eV]', 'Band gap (G₀W₀) [eV]',
  'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]',
  'Direct band gap (HSE06) [eV].1', 'CBM wrt. vacuum (PBE) [eV]',
  'VBM wrt. vacuum (PBE) [eV]'
], inplace=True)
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col].astype(str))
  label_encoders[col] = le

# Define target and features
target_col = 'Band gap (HSE06) [eV]'
df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]

# Impute and scale features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define MLP
class MLP(nn.Module):
  def __init__(self, input_size):
    super(MLP, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )

  def forward(self, x):
    return self.model(x)

# Instantiate model
model = MLP(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop with loss tracking
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  for batch_X, batch_y in train_loader:
    optimizer.zero_grad()
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  
  avg_loss = running_loss / len(train_loader)
  train_losses.append(avg_loss)

  model.eval()
  with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_loss = criterion(test_preds, y_test_tensor)
    test_losses.append(test_loss.item())

  if epoch % 10 == 0:
    print(f"Epoch {epoch}: Train MSE = {avg_loss:.4f}, Test MSE = {test_loss.item():.4f}")

# Final predictions
model.eval()
with torch.no_grad():
  y_pred = model(X_test_tensor).numpy().flatten()
  y_true = y_test_tensor.numpy().flatten()
  errors = y_pred - y_true

# Plot 1: Prediction vs Actual
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Actual Band Gap (HSE06) [eV]")
plt.ylabel("Predicted Band Gap (HSE06) [eV]")
plt.title("Predicted vs Actual Band Gap")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Training and Test Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Test Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Error Distribution
plt.figure(figsize=(6, 4))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel("Prediction Error (Pred - Actual)")
plt.title("Prediction Error Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Metrics
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Print the results
print(f"\nPerformance Metrics on Test Set:")
print(f"R² Score:    {r2:.4f}")
print(f"MAE:       {mae:.4f} eV")
print(f"RMSE:      {rmse:.4f} eV")
