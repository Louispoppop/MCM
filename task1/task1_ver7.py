"""
Task 1 Final Step: Inference of weekly vote shares using pre-calculated Base Popularity.

Input:
1.  (Judge scores & elimination results)
2.  (Base Popularity Shares from Version 7)

Output:
- task1_final_weekly_shares.csv (The ultimate estimated vote shares)
- Visualization of vote estimation certainty
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ==================== Config ====================
POPULARITY_FILE = "task1_global_popularity.csv"
DATA_FILE = "Cleaned_data.csv"
OUTPUT_FILE = "task1_final_weekly_shares.csv"

N_SAMPLES = 20000        # Sampling count for precision
# How much we trust the Base Popularity (higher = tighter spread)
PRIOR_STRENGTH = 50.0

# ==================== Helpers ====================


def load_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path = os.path.join(base_dir, DATA_FILE)
    pop_path = os.path.join(base_dir, POPULARITY_FILE)

    if not os.path.exists(data_path) or not os.path.exists(pop_path):
        raise FileNotFoundError(
            "Input files not found. Ensure both csvs exist.")

    df = pd.read_csv(data_path)
    df["total_judge_score"] = pd.to_numeric(
        df["total_judge_score"], errors="coerce")
    df = df[df["total_judge_score"].notna()].copy()

    pop_df = pd.read_csv(pop_path)
    return df, pop_df


def method_for_season(season: int) -> str:
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "rank_judges_save"


def compute_rank(scores: np.ndarray) -> np.ndarray:
    order = scores.argsort()[::-1]
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def is_consistent(judge: np.ndarray, fan: np.ndarray, elim: np.ndarray, method: str, placement: np.ndarray = None) -> bool:
    if placement is not None:
        # Final Week Logic
        if method == "rank":
            jr = compute_rank(judge)
            fr = compute_rank(fan)
            combined = jr + fr
            # Ascending combined score order (best to worst)
            pred_order = np.argsort(combined)
            pred_placements = placement[pred_order]
            return np.all(pred_placements[:-1] <= pred_placements[1:])

        elif method == "percent":
            jp = judge / judge.sum()
            combined = jp + fan
            # Descending combined score order (best to worst)
            pred_order = np.argsort(combined)[::-1]
            pred_placements = placement[pred_order]
            return np.all(pred_placements[:-1] <= pred_placements[1:])

        elif method == "rank_judges_save":
            jr = compute_rank(judge)
            fr = compute_rank(fan)
            combined = jr + fr
            pred_order = np.argsort(combined)
            pred_placements = placement[pred_order]
            return np.all(pred_placements[:-1] <= pred_placements[1:])

    if method == "rank":
        jr = compute_rank(judge)
        fr = compute_rank(fan)
        return elim[(jr + fr).argmax()]
    elif method == "percent":
        jp = judge / judge.sum()
        return elim[(jp + fan).argmin()]
    elif method == "rank_judges_save":
        jr = compute_rank(judge)
        fr = compute_rank(fan)
        return elim[(jr + fr).argsort()[::-1][:2]].any()
    return False

# ==================== Core Inference ====================


def infer_weekly_votes(df: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    results = []

    pop_map = {}
    for _, row in pop_df.iterrows():
        pop_map[(row["season"], row["celebrity_name"])
                ] = row["base_popularity_share"]

    groups = df.groupby(["season", "week"], sort=True)
    # Find max week per season for final detection
    season_max_weeks = df.groupby("season")["week"].max().to_dict()

    print(f"Running final inference on {len(groups)} weeks...")

    for (season, week), g in groups:
        season = int(season)
        week = int(week)
        method = method_for_season(season)

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        elim_mask = g["is_eliminated"].to_numpy(dtype=int).astype(bool)
        names = g["celebrity_name"].tolist()
        placements = g["placement"].to_numpy(dtype=float)

        is_final_week = (week == season_max_weeks[season])

        if elim_mask.sum() == 0 and not is_final_week:
            continue  # Skip non-elimination weeks (unless it's the final)

        # 1. Build Prior from Base Popularity
        base_shares = []
        default_share = 0.01  # Fallback

        for name in names:
            val = pop_map.get((season, name), default_share)
            base_shares.append(val)

        base_shares = np.array(base_shares)

        # 2. Re-normalize for current active contestants
        if base_shares.sum() > 0:
            active_prior_mean = base_shares / base_shares.sum()
        else:
            active_prior_mean = np.ones(len(names)) / len(names)

        # 3. Construct Dirichlet Alpha
        # Alpha = Mean * Strength
        alpha = active_prior_mean * PRIOR_STRENGTH

        # 4. Sampling
        candidates = rng.dirichlet(alpha, size=N_SAMPLES)

        # 5. Filter Consistent Samples
        feasible = []
        for fan_share in candidates:
            if is_consistent(judge_scores, fan_share, elim_mask, method, placements if is_final_week else None):
                feasible.append(fan_share)

        # 6. Calculate Stats
        if feasible:
            feasible = np.vstack(feasible)
            means = feasible.mean(axis=0)
            stds = feasible.std(axis=0)

            # 95% CI Width
            lower = np.percentile(feasible, 2.5, axis=0)
            upper = np.percentile(feasible, 97.5, axis=0)
            ci_widths = upper - lower

            for i, name in enumerate(names):
                results.append({
                    "season": season,
                    "week": week,
                    "celebrity_name": name,
                    "final_est_share": means[i],
                    "std_dev": stds[i],
                    "ci_width_95": ci_widths[i],
                    "judge_score": judge_scores[i],
                    "is_eliminated": elim_mask[i],
                    # The renormalized prior
                    "base_popularity": active_prior_mean[i]
                })
        else:
            # Fallback if no feasible solution found (rare with good prior)
            # Just return the prior mean as best guess
            for i, name in enumerate(names):
                results.append({
                    "season": season,
                    "week": week,
                    "celebrity_name": name,
                    "final_est_share": active_prior_mean[i],
                    "std_dev": np.nan,
                    "ci_width_95": np.nan,
                    "judge_score": judge_scores[i],
                    "is_eliminated": elim_mask[i],
                    "base_popularity": active_prior_mean[i]
                })

    return pd.DataFrame(results)


def plot_results(res_df: pd.DataFrame, base_dir: str):
    sns.set_theme(style="whitegrid")

    # Plot 1: Standard Deviation vs Mean Share (Certainty check)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=res_df, x="final_est_share",
                    y="std_dev", alpha=0.3, color="#8e44ad")
    plt.title("Constraint Tightness: Std Dev vs. Vote Share", fontsize=14)
    plt.xlabel("Estimated Vote Share", fontsize=12)
    plt.ylabel("Standard Deviation", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "final_inference_std_vs_share.png"))

    # Plot 2: Correlation between Judge Score and Fan Vote (Overall)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=res_df, x="judge_score",
                    y="final_est_share", alpha=0.2, color="#2980b9")
    plt.title("Correlation: Judge Score vs. Inferred Fan Vote", fontsize=14)
    plt.xlabel("Judge Score", fontsize=12)
    plt.ylabel("Fan Vote Share", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "final_inference_judge_vs_fan.png"))


def main():
    base_dir = os.path.dirname(__file__)

    try:
        df, pop_df = load_data(base_dir)
        res_df = infer_weekly_votes(df, pop_df)

        out_path = os.path.join(base_dir, OUTPUT_FILE)
        res_df.to_csv(out_path, index=False)
        print(f"Final estimates saved to {out_path}")

        plot_results(res_df, base_dir)

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
