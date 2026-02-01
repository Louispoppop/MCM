import pandas as pd
import numpy as np
import os

# Config
DATA_FILE = "Cleaned_data.csv"
ESTIMATES_FILE = "task1_ultimate_estimates.csv"
OUTPUT_FILE = "Cleaned_data_with_votes.csv"
TOTAL_VOTES_PER_WEEK = 14_000_000  # 14 Million hypothesis


def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, DATA_FILE)
    est_path = os.path.join(base_dir, ESTIMATES_FILE)

    if not os.path.exists(data_path) or not os.path.exists(est_path):
        print("Required files not found.")
        return

    # Load
    print("Loading data...")
    df_main = pd.read_csv(data_path)
    df_est = pd.read_csv(est_path)

    # Pre-processing keys to match types
    key_cols = ['season', 'week', 'celebrity_name']

    # Merge
    # We only want to add specific columns from estimates
    cols_to_add = ['final_est_share', 'std_dev',
                   'ci_width_95', 'base_popularity', 'perf_share']

    # Perform Merge
    print("Merging datasets...")
    merged = pd.merge(df_main, df_est[key_cols + cols_to_add],
                      on=key_cols,
                      how='left')

    # Calculate New Metrics
    print("Calculating metrics (CV & Vote Counts)...")

    # 1. Coefficient of Variation (CV)
    # Handle division by zero or very small numbers
    merged['vote_cv'] = merged['std_dev'] / merged['final_est_share']
    merged.loc[merged['final_est_share'] < 1e-6, 'vote_cv'] = 0.0  # Safety

    # 2. Vote Counts
    merged['vote_count_est'] = merged['final_est_share'] * TOTAL_VOTES_PER_WEEK
    # merged['vote_count_est'] = merged['vote_count_est'].fillna(0).astype(int)
    # rounding
    merged['vote_count_est'] = merged['vote_count_est'].round().astype('Int64')

    # Formatting
    # Move new columns to logical positions or just append
    # Let's verify merge success
    matched_count = merged['final_est_share'].notna().sum()
    print(f"Matched {matched_count} rows out of {len(merged)} total rows.")

    # Save
    out_path = os.path.join(base_dir, OUTPUT_FILE)
    merged.to_csv(out_path, index=False)
    print(f"Success! Saved merged data to: {OUTPUT_FILE}")



if __name__ == "__main__":
    main()
