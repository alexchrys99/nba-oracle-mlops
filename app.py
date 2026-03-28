import streamlit as st
import requests
import pandas as pd
import os
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NBA Oracle", page_icon="🏀", layout="wide")
st.title("🏀 NBA Live Oracle: Universal Radar")
st.markdown("### *AI-Powered Clutch Analysis & Momentum Tracking*")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    search_query = st.text_input("Enter Player Name (e.g., Luka Doncic, Stephen Curry):", "Stephen Curry")

@st.cache_data(ttl=3600)
def fetch_live_stats(player_name):
    try:
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict: return None, "Player not found."
        
        player_id = player_dict[0]['id']
        official_name = player_dict[0]['full_name']
        
        career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=30)
        df = career.get_data_frames()[0]
        if df.empty: return None, "No stats available."
        
        row = df.iloc[-1]
        return {
            "player_name": official_name,
            "MIN": float(row['MIN']), "PTS": float(row['PTS']),
            "REB": float(row['REB']), "AST": float(row['AST']),
            "TOV": float(row['TOV']), "FG_PCT": float(row['FG_PCT'])
        }, None
    except Exception as e:
        return None, str(e)

live_stats = None
if search_query:
    with st.spinner(f"📡 Searching NBA database..."):
        live_stats, error = fetch_live_stats(search_query)

with col2:
    if live_stats:
        # Create a Plotly Radar Chart
        categories = ['Points', 'Rebounds', 'Assists', 'Efficiency (FG%)', 'Ball Security (Inv. TOV)']
        
        # Normalize stats for the radar chart (simplistic scaling for visual comparison)
        player_values = [live_stats['PTS']/20, live_stats['REB']/5, live_stats['AST']/5, live_stats['FG_PCT']*2, 5 - live_stats['TOV']/2]
        avg_values = [15.0/20, 4.0/5, 3.0/5, 0.450*2, 5 - 2.0/2] # League average proxy
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=player_values, theta=categories, fill='toself', name=live_stats['player_name']))
        fig.add_trace(go.Scatterpolar(r=avg_values, theta=categories, fill='toself', name='League Average'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=True, margin=dict(t=20, b=20))
        
        st.plotly_chart(fig, use_container_width=True)
    elif search_query:
        st.error(error)

if live_stats:
    if st.button(f"🔮 Calculate True Clutch Impact for {live_stats['player_name']}", type="primary"):
        with st.spinner("🧠 Consulting the XGBoost and PyTorch Oracles..."):
            try:
                payload = {"opposing_lineup": [live_stats], "recent_plays": ["Made Shot", "Rebound", "Turnover"]}
                response = requests.post(f"{API_URL}/predict/defensive_strategy", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    score = result['target_clutch_score']
                    st.success("✅ Analysis Complete")
                    st.markdown(f"## 🎯 **{live_stats['player_name']}**")
                    if score > 30: st.error(f"### 🔥 LETHAL THREAT: `+{score:.2f}` Clutch Plus/Minus")
                    elif score > 10: st.warning(f"### ⚠️ HIGH THREAT: `+{score:.2f}` Clutch Plus/Minus")
                    else: st.info(f"### 🧊 COLD/LIABILITY: `{score:.2f}` Clutch Plus/Minus")
                else:
                    st.error("API Error")
            except:
                st.error(f"🚨 Could not connect to the API at {API_URL}.")
