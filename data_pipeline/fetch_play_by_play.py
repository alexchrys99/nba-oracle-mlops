import pandas as pd
import time
import os
import random
from nba_api.stats.endpoints import leaguegamelog, playbyplayv3

print("🎬 Initiating Smart Resume Play-by-Play Extraction...")

def fetch_with_retry(endpoint_class, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            response = endpoint_class(**kwargs, timeout=30)
            return response.get_data_frames()[0]
        except Exception as e:
            sleep_time = (3 ** attempt) + random.uniform(1.0, 3.0)
            print(f"   ⚠️ Connection dropped. Retrying in {sleep_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_time)
    return None

season = '2023-24'
all_game_ids = []
game_type_map = {}

print("Fetching Master Game List...")
try:
    rs_log = fetch_with_retry(leaguegamelog.LeagueGameLog, season=season, season_type_all_star='Regular Season')
    po_log = fetch_with_retry(leaguegamelog.LeagueGameLog, season=season, season_type_all_star='Playoffs')
except Exception as e:
    print(f"❌ Could not get Game IDs. {e}")
    exit()

for gid in rs_log['GAME_ID'].unique():
    all_game_ids.append(gid)
    game_type_map[gid] = 'Regular Season'
for gid in po_log['GAME_ID'].unique():
    all_game_ids.append(gid)
    game_type_map[gid] = 'Playoffs'

checkpoint_dir = "data_pipeline/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
processed_games = set()

print("🔍 Scanning checkpoints to see what we already downloaded...")
for file in os.listdir(checkpoint_dir):
    if file.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(checkpoint_dir, file))
        # THE FIX: Check for the new V3 'gameId' column!
        if 'GAME_ID' in df.columns:
            processed_games.update(df['GAME_ID'].unique())
        elif 'gameId' in df.columns:
            processed_games.update(df['gameId'].unique())

remaining_games = [gid for gid in all_game_ids if gid not in processed_games]

print(f"✅ Master List: {len(all_game_ids)} | Already Saved: {len(processed_games)} | Remaining: {len(remaining_games)}")

if len(remaining_games) == 0:
    print("🎉 All games downloaded! Ready to merge.")
else:
    print(f"🕰️ Resuming download for {len(remaining_games)} games. Using stealthier sleep timers...\n")

    pbp_data = []
    checkpoint_counter = len(os.listdir(checkpoint_dir)) + 1

    for i, game_id in enumerate(remaining_games):
        pbp = fetch_with_retry(playbyplayv3.PlayByPlayV3, max_retries=3, game_id=game_id)
        
        if pbp is not None and not pbp.empty:
            pbp['SEASON_TYPE'] = game_type_map[game_id] 
            
            # Standardize the column name before saving so we don't have this issue again
            if 'gameId' in pbp.columns:
                pbp.rename(columns={'gameId': 'GAME_ID'}, inplace=True)
                
            pbp_data.append(pbp)
            
            if (i + 1) % 50 == 0:
                temp_df = pd.concat(pbp_data, ignore_index=True)
                temp_df.to_parquet(f"{checkpoint_dir}/pbp_backup_resume_{checkpoint_counter}.parquet", engine='pyarrow')
                print(f"   💾 Checkpoint {checkpoint_counter} Saved: Processed {i+1} additional games.")
                pbp_data = [] 
                checkpoint_counter += 1
                
        time.sleep(2.5 + random.uniform(0, 2.0)) 

    # Save any remaining data that didn't hit a multiple of 50
    if pbp_data:
        temp_df = pd.concat(pbp_data, ignore_index=True)
        temp_df.to_parquet(f"{checkpoint_dir}/pbp_backup_resume_{checkpoint_counter}.parquet", engine='pyarrow')

# Final Merge 
print("\n🧬 Merging ALL checkpoints into the final Sequence Data Lake...")
final_chunks = []

for file in os.listdir(checkpoint_dir):
    if file.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(checkpoint_dir, file))
        # Ensure all historical V3 checkpoints are renamed for consistency
        if 'gameId' in df.columns:
             df.rename(columns={'gameId': 'GAME_ID'}, inplace=True)
        final_chunks.append(df)

if final_chunks:
    final_pbp_df = pd.concat(final_chunks, ignore_index=True)
    output_path = "data_pipeline/pbp_23_24_combined.parquet"
    final_pbp_df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"🏆 Ultimate Success! Sequence data merged and saved to {output_path} with {len(final_pbp_df)} total events.")
else:
    print("❌ Critical Error: No sequence data found to merge.")
