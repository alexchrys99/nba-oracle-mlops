import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pytorch
import itertools
import warnings
warnings.filterwarnings("ignore")

print("🧠 Booting up PyTorch Hyperparameter Grid Search...")

# 1. Load the Blood
df = pd.read_parquet("data_pipeline/clutch_veterans.parquet")
features = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
target = 'PLUS_MINUS'
df = df.dropna(subset=features + [target])

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# 2. Define the Upgraded Brain (Now with Dropout!)
class ClutchANN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(ClutchANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate) # Prevents memorization
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate) # Prevents memorization
        self.out = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.out(x)

# 3. The Hyperparameter Grid (12 Variations)
learning_rates = [0.01, 0.001]
hidden_sizes = [32, 64, 128]
dropout_rates = [0.2, 0.5]  # 20% or 50% neuron dropout
weight_decay = 1e-4         # L2 Regularization penalty
epochs = 200

# Set up MLflow
mlflow.set_experiment("NBA_Clutch_Impact_Predictor")

# 4. Run the Grid Search!
combinations = list(itertools.product(learning_rates, hidden_sizes, dropout_rates))
print(f"🔬 Testing {len(combinations)} different Neural Network architectures...\n")

for i, (lr, hs, drop) in enumerate(combinations):
    with mlflow.start_run(run_name=f"ANN_Var_{i+1}"):
        # Log parameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_size", hs)
        mlflow.log_param("dropout_rate", drop)
        mlflow.log_param("weight_decay", weight_decay)
        
        # Initialize model & optimizer
        model = ClutchANN(len(features), hs, drop)
        criterion = nn.MSELoss()
        # Adding weight_decay to Adam optimizer helps prevent overfitting
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_train_t)
            loss = criterion(predictions, y_train_t)
            loss.backward()
            optimizer.step()
            
            if epoch == epochs - 1:
                mlflow.log_metric("train_loss", loss.item())

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            mse = mean_squared_error(y_test, test_preds.numpy())
            mae = mean_absolute_error(y_test, test_preds.numpy())
            
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        
        print(f"Var {i+1:>2}/12 [LR: {lr:<5} | Hidden: {hs:<3} | Drop: {drop}] -> Test MAE: {mae:.2f} | Train/Test Gap: {mse - loss.item():.0f}")

print("\n✅ Grid search complete! Check the MLflow UI to find the Champion model.")
