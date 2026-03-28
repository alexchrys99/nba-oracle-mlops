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

print("🧠 Booting up Deep & Light Architecture Search...")

# 1. Load Data
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

# 2. Dynamic Deep Neural Network with Batch Normalization
class DynamicANN(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(DynamicANN, self).__init__()
        self.network = nn.Sequential()
        
        curr_size = input_size
        # Dynamically build layers based on the list provided
        for i, out_size in enumerate(hidden_layers):
            self.network.add_module(f"fc{i}", nn.Linear(curr_size, out_size))
            self.network.add_module(f"bn{i}", nn.BatchNorm1d(out_size))  # Stabilizes tabular learning
            self.network.add_module(f"relu{i}", nn.LeakyReLU(0.1))       # Prevents dead neurons
            self.network.add_module(f"drop{i}", nn.Dropout(dropout_rate))
            curr_size = out_size
            
        # Final Output Layer
        self.network.add_module("out", nn.Linear(curr_size, 1))

    def forward(self, x):
        return self.network(x)

# 3. The Architecture Grid
architectures = [
    ("Super_Light", [16, 8]),
    ("Deep_Light", [32, 16, 8]),
    ("Very_Deep_Light", [64, 32, 16, 8]),
    ("Deep_Wide", [128, 64, 32])
]
dropout_rates = [0.2, 0.4]
epochs = 200

mlflow.set_experiment("NBA_Clutch_Impact_Predictor")

# 4. Run the Show
combinations = list(itertools.product(architectures, dropout_rates))
print(f"🔬 Testing {len(combinations)} advanced architectures (Batch Norm + LeakyReLU)...\n")

for i, ((arch_name, layers), drop) in enumerate(combinations):
    with mlflow.start_run(run_name=f"DeepANN_{arch_name}_Drop{drop}"):
        
        mlflow.log_param("architecture", arch_name)
        mlflow.log_param("layers", str(layers))
        mlflow.log_param("dropout_rate", drop)
        
        model = DynamicANN(len(features), layers, drop)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_train_t)
            loss = criterion(predictions, y_train_t)
            loss.backward()
            optimizer.step()
            
            if epoch == epochs - 1:
                mlflow.log_metric("train_loss", loss.item())

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            mse = mean_squared_error(y_test, test_preds.numpy())
            mae = mean_absolute_error(y_test, test_preds.numpy())
            
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        
        print(f"Var {i+1:>2}/8 [{arch_name:<16} | Drop: {drop}] -> Test MAE: {mae:.2f}")

print("\n✅ Deep Grid Search complete!")
