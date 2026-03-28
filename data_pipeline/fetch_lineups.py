import pandas as pd
import time
import random
from nba_api.stats.endpoints import leaguedashlineups

print("🧠 Initiating 5-Man Lineup Synergy Extraction (Heavy Query Edition)...")

# Swapping to a Firefox User-Agent to rotate our fingerprint
custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com/',
    'Connection': 'keep-alive',
}

def fetch_with_retry(endpoint_class, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            # ⏳ INCREASED TIMEOUT TO 90 SECONDS
            response = endpoint_class(**kwargs, headers=custom_headers, timeout=90)
            return response.get_data_frames()[0]
        except Exception as e:
            sleep_time = (3 ** attempt) + random.uniform(2.0, 5.0)
            print(f"   ⚠️ Backend Error: {e}")
            print(f"   ⚠️ Retrying in {sleep_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_time)
    raise Exception("Failed after max retries. The NBA servers are refusing to answer.")

seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
all_lineups = []

for season in seasons:
    print(f"📥 Fetching Advanced 5-man lineups for {season} (This may take 60+ seconds)...")
    try:
        lineups = fetch_with_retry(
            leaguedashlineups.LeagueDashLineups, 
            season=season, 
            per_mode_detailed='Totals',
            measure_type_detailed_defense='Advanced'
        )
        
        df = lineups.copy()
        df['SEASON'] = season
        all_lineups.append(df)
        print(f"   ✅ Successfully pulled {season}!")
        
        time.sleep(3.0 + random.uniform(0, 2.0))
        
    except Exception as e:
        print(f"❌ Failed to fetch {season}: {e}")

if all_lineups:
    print("\n🧬 Merging all lineups into the Synergy Data Lake...")
    final_df = pd.concat(all_lineups, ignore_index=True)
    
    initial_count = len(final_df)
    df_filtered = final_df[final_df['MIN'] >= 40.0]
    
    print(f"📊 Filtered out 'garbage time' lineups: {initial_count} -> {len(df_filtered)} true rotational lineups.")
    
    output_path = "data_pipeline/lineups_5yr.parquet"
    df_filtered.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"🏆 Success! Lineup data locked and loaded at: {output_path}")
else:
    print("❌ Critical Error: No lineup data was downloaded.")
