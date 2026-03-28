import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashplayerclutch

print("🏀 Initiating 30-Year Historical NBA Data Pipeline (1996 - Present)...")

# Dynamically generate season strings (e.g., '1996-97', '1999-00', '2000-01')
seasons = []
for year in range(1996, 2026):
    next_year = str(year + 1)[-2:]
    seasons.append(f"{year}-{next_year}")

all_seasons_data = []

print(f"🕰️ Preparing to download {len(seasons)} seasons of clutch data. This will take ~1-2 minutes to avoid IP bans...\n")

for season in seasons:
    print(f"📥 Fetching: {season}...", end=" ")
    try:
        clutch_stats = leaguedashplayerclutch.LeagueDashPlayerClutch(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='Totals'
        )
        
        df = clutch_stats.get_data_frames()[0]
        df['SEASON'] = season  
        all_seasons_data.append(df)
        print(f"✅ Found {len(df)} players.")
        
        # ⚠️ CRITICAL: Sleep for 2.5 seconds to respect NBA servers
        time.sleep(2.5)
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if all_seasons_data:
    print("\n🧬 Merging 30 years of NBA history into a single dataset...")
    final_df = pd.concat(all_seasons_data, ignore_index=True)

    # Apply the Master Thesis Filter: >= 20 Games AND >= 60 Minutes
    print("⚖️ Filtering for reliable veterans (>= 20 clutch games AND >= 60 clutch minutes)...")
    df_filtered = final_df[(final_df['GP'] >= 20) & (final_df['MIN'] >= 60)]

    print(f"📊 Dataset exploded from 872 rows to {len(df_filtered)} true clutch veteran seasons!")

    output_path = "data_pipeline/clutch_veterans.parquet"
    df_filtered.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"💾 Success! Historical data locked and loaded at: {output_path}")
else:
    print("❌ Critical Error: No data was downloaded.")
