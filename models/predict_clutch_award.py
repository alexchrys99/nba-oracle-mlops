import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🏆 Initiating the NBA Oracle Award Committee...")

# 1. Load the 30-Year Database
df = pd.read_parquet("data_pipeline/clutch_veterans.parquet")

# 2. Engineer the exact same features the AI was trained on
df['PTS_PER_MIN'] = df['PTS'] / df['MIN']
df['REB_PER_MIN'] = df['REB'] / df['MIN']
df['AST_PER_MIN'] = df['AST'] / df['MIN']
df['STL_PER_MIN'] = df['STL'] / df['MIN']
df['BLK_PER_MIN'] = df['BLK'] / df['MIN']
df['AST_TO_TOV'] = df['AST'] / (df['TOV'] + 0.1) 

features = ['MIN', 'PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN', 'STL_PER_MIN', 'BLK_PER_MIN', 'AST_TO_TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
df = df.dropna(subset=features + ['PLUS_MINUS'])

# 3. Train the Champion AI on historical data (Everything BEFORE 2023-24)
historical_data = df[df['SEASON'] != '2023-24']
current_season = df[df['SEASON'] == '2023-24'].copy()

print(f"🧠 Training AI on {len(historical_data)} historical clutch seasons...")
xgb_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42, n_jobs=-1)
xgb_model.fit(historical_data[features], historical_data['PLUS_MINUS'])

# 4. Predict the True Clutch Impact for the 2023-24 Season
print(f"🔮 Evaluating {len(current_season)} players from the 2023-24 season...")
current_season['AI_CLUTCH_SCORE'] = xgb_model.predict(current_season[features])

# 5. Rank the candidates!
top_candidates = current_season.sort_values(by='AI_CLUTCH_SCORE', ascending=False).head(5)

print("\n" + "="*60)
print("🏆 ORACLE'S 2023-24 CLUTCH PLAYER OF THE YEAR BALLOT 🏆")
print("="*60)

rank = 1
for index, row in top_candidates.iterrows():
    print(f"#{rank} | {row['PLAYER_NAME']}")
    print(f"    ↳ AI Projected Impact: +{row['AI_CLUTCH_SCORE']:.2f} Points")
    print(f"    ↳ Actual Stats: {row['PTS']} PTS, {row['AST']} AST, {row['FG_PCT']*100:.1f}% FG in {row['MIN']:.1f} Clutch Minutes")
    print("-" * 60)
    rank += 1
