import pandas as pd
import time
import random
from nba_api.stats.endpoints import leaguehustlestatsplayer

print("🛡️ Initiating Defensive Hustle Extraction (Patient Edition)...")

custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com/',
}

def fetch_with_retry(endpoint_class, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            # ⏳ INCREASED TIMEOUT TO 60 SECONDS
            response = endpoint_class(**kwargs, headers=custom_headers, timeout=60)
            return response.get_data_frames()[0]
        except Exception as e:
            sleep_time = (2 ** attempt) + random.uniform(1.0, 3.0)
            print(f"   ⚠️ Server taking too long. Retrying in {sleep_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_time)
    raise Exception("Failed after max retries. The NBA servers are just too slow right now.")

season = '2023-24'

print(f"📥 Fetching Hustle Stats for {season} (Allow up to 60 seconds)...")
try:
    df = fetch_with_retry(
        leaguehustlestatsplayer.LeagueHustleStatsPlayer,
        season=season, 
        per_mode_time='Totals'
    )
    
    # Filter for players who actually play
    df_filtered = df[df['MIN'] >= 500]
    print(f"✅ Success! Pulled {len(df_filtered)} rotational defensive players.")
    
    output_path = "data_pipeline/hustle_23_24.parquet"
    df_filtered.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"💾 Defensive Data locked and loaded at: {output_path}")

except Exception as e:
    print(f"❌ Error: {e}")
