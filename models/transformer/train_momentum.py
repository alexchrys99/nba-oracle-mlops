import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import warnings
warnings.filterwarnings("ignore")

print("🧠 Booting up the PyTorch Sequence Transformer...")

# 1. Load the Sequence Data
df = pd.read_parquet("data_pipeline/pbp_23_24_combined.parquet")

if 'actionType' in df.columns:
    df.rename(columns={'actionType': 'EVENTMSGTYPE'}, inplace=True)
if 'gameId' in df.columns:
    df.rename(columns={'gameId': 'GAME_ID'}, inplace=True)

# 🛠️ THE MLOPS FIX: Tokenization!
# Ensure everything is a string, and drop missing values
df = df.dropna(subset=['EVENTMSGTYPE'])
df['EVENTMSGTYPE'] = df['EVENTMSGTYPE'].astype(str)

# Build a dynamic dictionary mapping string events to unique integers
unique_events = df['EVENTMSGTYPE'].unique()
vocab_size = len(unique_events)
event_to_id = {event: idx for idx, event in enumerate(unique_events)}

print(f"📖 Built a vocabulary of {vocab_size} unique basketball actions.")
print("   Sample tokens:", list(event_to_id.items())[:4])

# Apply the tokenization to the dataset
df['EVENT_TOKEN'] = df['EVENTMSGTYPE'].map(event_to_id)
print(f"🧹 Cleaned dataset down to {len(df):,} core basketball plays.")

# 2. Build the Sequences (The "Sentences" of Basketball)
SEQUENCE_LENGTH = 10  

def create_sequences(dataframe, seq_length):
    sequences = []
    targets = []
    for game_id, group in dataframe.groupby('GAME_ID'):
        events = group['EVENT_TOKEN'].values
        for i in range(len(events) - seq_length):
            seq = events[i : i + seq_length]
            target = events[i + seq_length]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)

print(f"✂️ Chopping games into {SEQUENCE_LENGTH}-play rolling windows...")
X, y = create_sequences(df, SEQUENCE_LENGTH)
print(f"✅ Created {len(X):,} training sequences.")

# Train/Test Split
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

X_train_t = torch.LongTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.LongTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 3. Define the AI Brain
class NBATransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):
        super(NBATransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_out = self.transformer(embedded)
        last_hidden_state = transformer_out[:, -1, :]
        out = self.fc_out(last_hidden_state)
        return out

model = NBATransformer(vocab_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

mlflow.set_experiment("NBA_Momentum_Transformer")

with mlflow.start_run(run_name="Transformer_V1"):
    mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
    mlflow.log_param("epochs", epochs)
    
    print("🚀 Training the Transformer... (This might take a minute)")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        mlflow.log_metric("train_loss", total_loss/len(train_loader), step=epoch)

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        predicted_classes = torch.argmax(test_preds, dim=1)
        correct = (predicted_classes == y_test_t).sum().item()
        total = y_test_t.size(0)

    accuracy = correct / total
    print("-" * 50)
    print(f"🎯 Next-Play Prediction Accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)
    mlflow.log_metric("test_accuracy", accuracy)

print("✅ Transformer pipeline complete!")
