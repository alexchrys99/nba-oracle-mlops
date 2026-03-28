import pandas as pd
import os

print("🧬 Bypassing the Firewall: Merging the games we successfully secured...")

checkpoint_dir = "data_pipeline/checkpoints"
final_chunks = []

print("🔍 Scanning local checkpoints...")
for file in os.listdir(checkpoint_dir):
    if file.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(checkpoint_dir, file))
        
        # Ensure all columns match before the merge
        if 'gameId' in df.columns:
             df.rename(columns={'gameId': 'GAME_ID'}, inplace=True)
             
        final_chunks.append(df)

if final_chunks:
    print("🔨 Forging the final Data Lake...")
    final_pbp_df = pd.concat(final_chunks, ignore_index=True)
    
    # Sort chronologically just to be safe
    # Play-by-play events usually have an EVENTNUM we can sort by
    if 'EVENTNUM' in final_pbp_df.columns:
        final_pbp_df = final_pbp_df.sort_values(by=['GAME_ID', 'EVENTNUM'])
        
    output_path = "data_pipeline/pbp_23_24_combined.parquet"
    final_pbp_df.to_parquet(output_path, engine='pyarrow', index=False)
    
    print("="*60)
    print(f"🏆 Ultimate Success! Phase 4 Complete.")
    print(f"📊 Final Dataset Size: {len(final_pbp_df):,} chronological events.")
    print(f"💾 Saved to: {output_path}")
    print("="*60)
else:
    print("❌ Critical Error: No sequence data found in the checkpoints folder.")
