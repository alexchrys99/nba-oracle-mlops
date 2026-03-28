import streamlit as st
import requests
import pandas as pd
import os
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

# The MLOps API Router (Allows it to work locally OR inside Docker)
API_URL = os.getenv("API_URL", "http://nba-oracle:8000")

st.set_page_config(page_title="NBA Oracle", page_icon="🏀", layout="wide")
st.title("🏀 NBA Live Oracle: Universal Radar")
st.markdown("### *Search for any player to fetch their live stats and calculate True Clutch Impact.*")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Universal Player Search")
    # THE UPGRADE: A free-text search box
    search_query = st.text_input("Enter Player Name (e.g., LeBron James, Tyrese Maxey):", "LeBron James")

@st.cache_data(ttl=3600)
def fetch_live_stats(player_name):
    try:
        # 1. Search the NBA database for the string the user typed
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict: return None, "Player not found. Check your spelling."
        
        # Grab the exact official name and ID of the first match
        player_id = player_dict[0]['id']
        official_name = player_dict[0]['full_name']
        
        career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=30)
        df = career.get_data_frames()[0]
        if df.empty: return None, "Player found, but no stats available."
        
        recent_season = df.iloc[-1:] 
        row = recent_season.iloc[0]
        
        return {
            "player_name": official_name,
            "MIN": float(row['MIN']),
            "PTS": float(row['PTS']),
            "REB": float(row['REB']),
            "AST": float(row['AST']),
            "TOV": float(row['TOV']),
            "FG_PCT": float(row['FG_PCT'])
        }, None
    except Exception as e:
        return None, str(e)

live_stats = None
if search_query:
    with st.spinner(f"📡 Searching NBA database for '{search_query}'..."):
        live_stats, error = fetch_live_stats(search_query)

with col2:
    st.subheader("📊 Current Season Averages (Totals)")
    if live_stats:
        st.success(f"✅ Found Official Record: **{live_stats['player_name']}**")
        st.json(live_stats)
    elif search_query:
        st.error(error)

st.divider()

if live_stats:
    if st.button(f"🔮 Calculate True Clutch Impact for {live_stats['player_name']}", type="primary", use_container_width=True):
        with st.spinner("🧠 The XGBoost Oracle is analyzing the math..."):
            try:
                payload = {
                    "opposing_lineup": [live_stats],
                    "recent_plays": ["Made Shot", "Rebound", "Turnover"] 
                }
                
                # Send to our dynamic API URL
                response = requests.post(f"{API_URL}/predict/defensive_strategy", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    clutch_score = result['target_clutch_score']
                    
                    st.success("✅ Analysis Complete")
                    st.markdown(f"## 🎯 **{live_stats['player_name']}**")
                    
                    if clutch_score > 30:
                        st.error(f"### 🔥 LETHAL THREAT: `+{clutch_score:.2f}` Clutch Plus/Minus")
                    elif clutch_score > 10:
                        st.warning(f"### ⚠️ HIGH THREAT: `+{clutch_score:.2f}` Clutch Plus/Minus")
                    else:
                        st.info(f"### 🧊 COLD/LIABILITY: `{clutch_score:.2f}` Clutch Plus/Minus")
                else:
                    st.error(f"API Error {response.status_code}")
            except Exception as e:
                st.error(f"🚨 Could not connect to the API at {API_URL}.")
