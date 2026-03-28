import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pytorch
import warnings
warnings.filterwarnings("ignore")

print("🧠 Booting up NBA Oracle ML Pipeline...")

# 1. Load the Blood (Data)
data_path = "data_pipeline/clutch_veterans.parquet"
df = pd.read_parquet(data_path)

# 2. Features & Target
# We predict PLUS_MINUS (Winning Impact) based on raw box score stats
features = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
target = 'PLUS_MINUS'

# Drop any rows with missing data
df = df.dropna(subset=features + [target])

X = df[features].values
y = df[target].values

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (Neural Networks love numbers between -1 and 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert arrays into PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# 3. Define the Brain (PyTorch Multi-Layer Perceptron)
class ClutchANN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClutchANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

# Hyperparameters
input_size = len(features)
hidden_size = 64
learning_rate = 0.01
epochs = 200

model = ClutchANN(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. MLOps: Track everything with MLflow
mlflow.set_experiment("NBA_Clutch_Impact_Predictor")

with mlflow.start_run():
    print(f"🚀 Training PyTorch ANN with {epochs} epochs...")
    
    # Log our architecture settings
    mlflow.log_param("hidden_size", hidden_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("features", ", ".join(features))

    # The Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"Epoch [{(epoch+1):>3}/{epochs}] | Training Loss (MSE): {loss.item():.4f}")
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

    # 5. Evaluate the Model on Unseen Data
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        mse = mean_squared_error(y_test, test_preds.numpy())
        mae = mean_absolute_error(y_test, test_preds.numpy())

    print("-" * 40)
    print(f"🎯 Final Test MSE: {mse:.4f}")
    print(f"🎯 Final Test MAE: {mae:.4f} (Average error in Plus/Minus prediction)")
    
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mae", mae)

    # 6. Save the Champion Model to the Registry
    mlflow.pytorch.log_model(model, "model")
    print("💾 Model weights successfully saved to MLflow!")
