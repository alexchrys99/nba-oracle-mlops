import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import itertools
import warnings
warnings.filterwarnings("ignore")

print("🌲 Booting up XGBoost REGRESSOR (Advanced Feature Engineering)...")

# 1. Load Data
df = pd.read_parquet("data_pipeline/clutch_veterans.parquet")

# 2. FEATURE ENGINEERING (Pivot 2)
print("🧪 Engineering 'Per Minute' and 'Ratio' metrics...")
df['PTS_PER_MIN'] = df['PTS'] / df['MIN']
df['REB_PER_MIN'] = df['REB'] / df['MIN']
df['AST_PER_MIN'] = df['AST'] / df['MIN']
df['STL_PER_MIN'] = df['STL'] / df['MIN']
df['BLK_PER_MIN'] = df['BLK'] / df['MIN']
# Avoid division by zero for Turnovers
df['AST_TO_TOV'] = df['AST'] / (df['TOV'] + 0.1) 

# We drop the raw stats and use our engineered rate stats!
features = ['MIN', 'PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN', 'STL_PER_MIN', 'BLK_PER_MIN', 'AST_TO_TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
target = 'PLUS_MINUS'

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("NBA_Clutch_Impact_Predictor")

# 3. XGBoost Hyperparameter Grid (12 Variations)
max_depths = [3, 5]
learning_rates = [0.01, 0.05, 0.1]
n_estimators = [100, 200]

combinations = list(itertools.product(max_depths, learning_rates, n_estimators))
print(f"🔬 Testing {len(combinations)} XGBoost variations on engineered features...\n")

best_mae = float("inf")
best_model = None
best_features = None

for i, (depth, lr, estimators) in enumerate(combinations):
    with mlflow.start_run(run_name=f"XGB_Eng_d{depth}_lr{lr}_n{estimators}"):
        
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("n_estimators", estimators)
        
        # Build & Train the Model (Regression)
        model = xgb.XGBRegressor(
            max_depth=depth, 
            learning_rate=lr, 
            n_estimators=estimators, 
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_mse", mse)
        
        print(f"Var {i+1:>2}/12 [Depth: {depth} | LR: {lr:<4} | Trees: {estimators:<3}] -> Test MAE: {mae:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = model

# 4. Feature Importance
print("\n" + "="*50)
print(f"🏆 BEST MAE ACHIEVED: {best_mae:.2f}")
print("="*50)
importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n📊 What actually drives Plus/Minus? (Feature Importances):")
for i, row in importance_df.head(5).iterrows():
    print(f"   - {row['Feature']}: {row['Importance']*100:.1f}%")
