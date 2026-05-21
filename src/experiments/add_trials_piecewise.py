import pandas as pd
import os

# Define the base paths
DATASET = "summeval"
source_base = f"/Users/alyssaunell/code/SmartSample_local/results/05_03_VM_{DATASET}_alyssa_30_pairwise_average/{DATASET}/dataframes"
dest_base = f"/Users/alyssaunell/code/SmartSample_local/results/05_03_VM_{DATASET}_alyssa_40_pairwise_average/{DATASET}/dataframes"

# Define the metrics
metrics = ['alpha', 'icc', 'mse', 'rho', 'tau']

# Process each metric
for metric in metrics:
    source_file = os.path.join(source_base, f"{metric}_results.csv")
    dest_file = os.path.join(dest_base, f"{metric}_results.csv")
    
    # Read the source CSV
    source_df = pd.read_csv(source_file)
    print(f"\nProcessing {metric}_results.csv")
    print(f"  Source rows: {len(source_df)}")
    
    # Read the destination CSV
    dest_df = pd.read_csv(dest_file)
    print(f"  Destination rows (before): {len(dest_df)}")
    
    # Append source to destination
    combined_df = pd.concat([dest_df, source_df], ignore_index=True)
    print(f"  Destination rows (after): {len(combined_df)}")
    
    # Save back to destination file
    combined_df.to_csv(dest_file, index=False)
    print(f"  ✓ Saved to {dest_file}")

print("\n✓ All files have been updated successfully!")