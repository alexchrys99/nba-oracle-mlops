from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import xgboost as xgb
import pandas as pd
import torch
import torch.nn as nn
import uvicorn

app = FastAPI(title="🏀 NBA Defensive Oracle", description="Tactical Double-Team Recommender")

print("⏳ Booting up XGBoost Clutch Brain...")
df = pd.read_parquet("data_pipeline/clutch_veterans.parquet")
df['PTS_PER_MIN'] = df['PTS'] / df['MIN']
df['REB_PER_MIN'] = df['REB'] / df['MIN']
df['AST_PER_MIN'] = df['AST'] / df['MIN']
df['STL_PER_MIN'] = df['STL'] / df['MIN']
df['BLK_PER_MIN'] = df['BLK'] / df['MIN']
df['AST_TO_TOV'] = df['AST'] / (df['TOV'] + 0.1) 
features = ['MIN', 'PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN', 'STL_PER_MIN', 'BLK_PER_MIN', 'AST_TO_TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
df = df.dropna(subset=features + ['PLUS_MINUS'])
xgb_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42, n_jobs=-1)
xgb_model.fit(df[features], df['PLUS_MINUS'])

print("⏳ Booting up PyTorch Momentum Brain...")
vocab = {0: 'period', 1: 'Jump Ball', 2: 'Made Shot', 3: 'Missed Shot', 4: 'Rebound', 5: 'Turnover', 6: 'Foul', 7: 'Violation', 8: 'Substitution', 9: 'Timeout', 10: 'Free Throw'}
reverse_vocab = {v: k for k, v in vocab.items()}

class NBATransformer(nn.Module):
    def __init__(self, vocab_size=15, d_model=32, nhead=4, num_layers=2):
        super(NBATransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, batch_first=True), num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return self.fc(self.transformer(self.embedding(x))[:, -1, :])

torch_model = NBATransformer()
torch_model.eval()
print("✅ Master Defensive Oracle is LIVE!")

# The new payload expects a LIST of players to compare against each other
class PlayerStats(BaseModel):
    player_name: str
    MIN: float
    PTS: float
    REB: float
    AST: float
    TOV: float
    FG_PCT: float

class DefensiveStrategyRequest(BaseModel):
    opposing_lineup: List[PlayerStats]
    recent_plays: List[str]

@app.post("/predict/defensive_strategy")
def predict_strategy(req: DefensiveStrategyRequest):
    try:
        # 1. XGBoost: Rank the opponents by clutch lethality
        threat_rankings = []
        for p in req.opposing_lineup:
            pts_pm, reb_pm, ast_pm = p.PTS / p.MIN, p.REB / p.MIN, p.AST / p.MIN
            ast_to_tov = p.AST / (p.TOV + 0.1)
            xgb_input = pd.DataFrame([[p.MIN, pts_pm, reb_pm, ast_pm, 0.02, 0.01, ast_to_tov, p.FG_PCT, 0.35, 0.80]], columns=features)
            clutch_score = float(xgb_model.predict(xgb_input)[0])
            threat_rankings.append({"player": p.player_name, "threat_score": clutch_score})
            
        # Sort to find the most dangerous player
        threat_rankings.sort(key=lambda x: x["threat_score"], reverse=True)
        primary_target = threat_rankings[0]

        # 2. PyTorch: Predict the next likely event to anticipate the play type
        sequence_ids = [reverse_vocab.get(play, 3) for play in req.recent_plays]
        if len(sequence_ids) < 10: sequence_ids = ([0] * (10 - len(sequence_ids))) + sequence_ids
        else: sequence_ids = sequence_ids[-10:]
        
        with torch.no_grad():
            tensor_input = torch.LongTensor([sequence_ids])
            prediction = torch_model(tensor_input)
            predicted_id = torch.argmax(prediction, dim=1).item()
            next_play = vocab.get(predicted_id, "Unknown Play")

        # 3. Formulate the tactical advice
        tactical_advice = f"Send the double team to {primary_target['player']}. Get the ball out of their hands immediately. They have the highest clutch projection (+{primary_target['threat_score']:.2f})."
        
        if next_play in ['Made Shot', 'Free Throw', 'Foul']:
            tactical_advice += f" HIGH ALERT: Momentum indicates the offense is aggressively attacking (Predicted next event: {next_play}). Trap the pick-and-roll early."
        else:
            tactical_advice += f" Anticipate chaos: Predicted next event is {next_play}."

        return {
            "primary_target": primary_target['player'],
            "target_clutch_score": round(primary_target['threat_score'], 2),
            "predicted_momentum_event": next_play,
            "tactical_recommendation": tactical_advice,
            "all_threats": threat_rankings
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
