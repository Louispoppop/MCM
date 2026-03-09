import os
import sys
import numpy as np
import pandas as pd

INPUT = r"d:\美赛\task1_estimates.csv"
OUT_ROW = r"d:\美赛\task1_estimates_with_cv.csv"
OUT_WEEK = r"d:\美赛\task1_week_cv_summary.csv"

df = pd.read_csv(INPUT)

# Ensure numeric
df["est_fan_vote_mean"] = pd.to_numeric(
    df["est_fan_vote_mean"], errors="coerce")
df["est_fan_vote_std"] = pd.to_numeric(df["est_fan_vote_std"], errors="coerce")

# CV = std / mean; handle mean <= 0
cv = df["est_fan_vote_std"] / df["est_fan_vote_mean"]
cv = cv.replace([np.inf, -np.inf], np.nan)
cv[df["est_fan_vote_mean"] <= 0] = np.nan

df["cv"] = cv

# Week-level summary
week_summary = (
    df.groupby(["season", "week"], sort=True)
    .agg(
        n_candidates=("celebrity_name", "count"),
        mean_cv=("cv", "mean"),
        median_cv=("cv", "median"),
        pct_high_confidence=("cv", lambda x: np.nanmean(x < 0.1)),
        pct_medium_confidence=(
            "cv", lambda x: np.nanmean((x >= 0.1) & (x < 0.3))),
        pct_low_confidence=("cv", lambda x: np.nanmean(x >= 0.3)),
    )
    .reset_index()
)

# Save outputs
df.to_csv(OUT_ROW, index=False)
week_summary.to_csv(OUT_WEEK, index=False)

print("Saved:", OUT_ROW, OUT_WEEK)
