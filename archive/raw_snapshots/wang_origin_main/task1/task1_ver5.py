"""
Task 1: Estimate fan vote shares using rejection-sampling Monte Carlo.
Combined script with estimation, uncertainty analysis (CV), and visualization.
Final version with enhanced visuals.

Inputs:
    - Cleaned_data.csv

Outputs:
    - task1_estimates_complete.csv
    - task1_week_consistency.csv
    - Analysis plots (png files)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# Data Structures & Helpers
# ==========================================

@dataclass
class WeekResult:
    season: int
    week: int
    method: str
    feasible_rate: float
    reproduced: bool
    pred_success: bool


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)
    df = df.copy()
    df["total_judge_score"] = pd.to_numeric(
        df["total_judge_score"], errors="coerce")
    # Keep only active rows (those with valid judge scores)
    df = df[df["total_judge_score"].notna()]
    return df


def method_for_season(season: int) -> str:
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "rank_judges_save"


def compute_judge_rank(scores: np.ndarray) -> np.ndarray:
    # Higher score => better rank (1 is best)
    order = scores.argsort()[::-1]
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def compute_fan_rank(votes: np.ndarray) -> np.ndarray:
    order = votes.argsort()[::-1]
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(votes) + 1)
    return ranks


def is_elimination_consistent(
    judge_scores: np.ndarray,
    fan_share: np.ndarray,
    eliminated_mask: np.ndarray,
    method: str
) -> bool:
    """Check if a given fan vote share leads to the actual elimination outcome."""
    if method == "rank":
        judge_rank = compute_judge_rank(judge_scores)
        fan_rank = compute_fan_rank(fan_share)
        combined = judge_rank + fan_rank
        eliminated_pred = combined.argmax()
        return eliminated_mask[eliminated_pred]

    elif method == "percent":
        judge_percent = judge_scores / judge_scores.sum()
        combined = judge_percent + fan_share
        eliminated_pred = combined.argmin()
        return eliminated_mask[eliminated_pred]

    elif method == "rank_judges_save":
        judge_rank = compute_judge_rank(judge_scores)
        fan_rank = compute_fan_rank(fan_share)
        combined = judge_rank + fan_rank
        bottom_two = combined.argsort()[::-1][:2]
        return eliminated_mask[bottom_two].any()

    return False


# ==========================================
# Simulation Logic
# ==========================================

def simulate_week(
    judge_scores: np.ndarray,
    eliminated_mask: np.ndarray,
    method: str,
    n_samples: int = 10000,
    alpha_scale: float = 0.8,
    lognormal_sigma: float = 0.6,
    rng: np.random.Generator | None = None,
    fan_prior: str = "dirichlet_uniform",
) -> Tuple[np.ndarray, int]:

    if rng is None:
        rng = np.random.default_rng(42)

    n = len(judge_scores)

    def sample_fan_share() -> np.ndarray:
        if fan_prior == "dirichlet_uniform":
            return rng.dirichlet(np.ones(n))
        elif fan_prior == "dirichlet_concentrated":
            return rng.dirichlet(np.ones(n) * alpha_scale)
        elif fan_prior == "lognormal":
            x = rng.lognormal(mean=0.0, sigma=lognormal_sigma, size=n)
            return x / x.sum()
        elif fan_prior == "normal_softmax":
            x = rng.normal(loc=0.0, scale=1.0, size=n)
            e = np.exp(x - x.max())
            return e / e.sum()
        else:
            raise ValueError(f"Unknown fan_prior: {fan_prior}")

    feasible = []
    total_feasible = 0

    # Batch generation for speed could be implemented, but loop is clearer for logic
    for _ in range(n_samples):
        fan_share = sample_fan_share()
        if is_elimination_consistent(judge_scores, fan_share, eliminated_mask, method):
            feasible.append(fan_share)
            total_feasible += 1

    if feasible:
        return np.vstack(feasible), total_feasible
    return np.empty((0, n)), total_feasible


def estimate_fan_votes(
    df: pd.DataFrame,
    n_samples: int = 20000,
    alpha_scale: float = 8.0,
    lognormal_sigma: float = 0.6,
    seed: int = 42,
    fan_prior: str = "dirichlet_concentrated",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)
    estimates = []
    week_results: List[WeekResult] = []

    groups = df.groupby(["season", "week"], sort=True)
    total_groups = len(groups)
    print(f"Processing {total_groups} weeks...")

    for (season, week), g in groups:
        g = g.copy()
        method = method_for_season(int(season))

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        eliminated_mask = g["is_eliminated"].to_numpy(dtype=int).astype(bool)

        if eliminated_mask.sum() == 0:
            continue

        feasible_votes, total_feasible = simulate_week(
            judge_scores=judge_scores,
            eliminated_mask=eliminated_mask,
            method=method,
            n_samples=n_samples,
            alpha_scale=alpha_scale,
            lognormal_sigma=lognormal_sigma,
            rng=rng,
            fan_prior=fan_prior,
        )

        feasible_rate = total_feasible / n_samples
        reproduced = total_feasible > 0

        # Basic entries for rows
        names = g["celebrity_name"].tolist()

        if feasible_votes.size == 0:
            week_results.append(WeekResult(int(season), int(
                week), method, feasible_rate, reproduced, False))
            for i, name in enumerate(names):
                estimates.append({
                    "season": int(season),
                    "week": int(week),
                    "celebrity_name": name,
                    "est_fan_vote_mean": np.nan,
                    "est_fan_vote_median": np.nan,
                    "est_fan_vote_std": np.nan,
                    "feasible_samples": 0,
                    "cv": np.nan,
                    "judge_score": float(judge_scores[i])
                })
        else:
            mean = feasible_votes.mean(axis=0)
            median = np.median(feasible_votes, axis=0)
            std = feasible_votes.std(axis=0)

            # Prediction success check (Mean-based)
            pred_success = is_elimination_consistent(
                judge_scores, mean, eliminated_mask, method
            )
            week_results.append(WeekResult(int(season), int(
                week), method, feasible_rate, reproduced, bool(pred_success)))

            for i, (name, m, md, s) in enumerate(zip(names, mean, median, std)):
                # Calculate CV (Coefficient of Variation) = std / mean
                cv_val = s / m if m > 1e-9 else np.nan

                estimates.append({
                    "season": int(season),
                    "week": int(week),
                    "celebrity_name": name,
                    "est_fan_vote_mean": float(m),
                    "est_fan_vote_median": float(md),
                    "est_fan_vote_std": float(s),
                    "feasible_samples": int(total_feasible),
                    "cv": float(cv_val),
                    "judge_score": float(judge_scores[i])
                })

    return pd.DataFrame(estimates), pd.DataFrame([wr.__dict__ for wr in week_results])


# ==========================================
# Visualization
# ==========================================

def plot_analysis(estimates_df: pd.DataFrame, week_df: pd.DataFrame, save_dir: str):
    sns.set_theme(style="whitegrid")

    # Plot 1: Histogram of CV (Certainty Distribution)
    plt.figure(figsize=(10, 6))
    clean_cv = estimates_df["cv"].dropna()
    sns.histplot(clean_cv, bins=50, kde=True,
                 color="#66c2a5", edgecolor="black")
    plt.title(
        "Distribution of Coefficient of Variation (CV)\n(Uncertainty)", fontsize=14)
    plt.xlabel("Coefficient of Variation (std / mean)", fontsize=12)
    plt.ylabel("Count of Celebrity-Weeks", fontsize=12)
    plt.axvline(x=0.1, color='green', linestyle='--',
                alpha=0.7, label='High Certainty (<0.1)')
    plt.axvline(x=0.3, color='orange', linestyle='--',
                alpha=0.7, label='Low Certainty (>0.3)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "analysis_cv_histogram.png"), dpi=300)
    plt.close()

    # Plot 2: Average CV by Season (Bar Chart) - Uncertainty "Over Time"
    if not estimates_df.empty:
        season_cv = estimates_df.groupby("season")["cv"].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=season_cv, x="season", y="cv", color="#8da0cb")
        plt.title("Average Fan Vote Uncertainty (CV) by Season", fontsize=14)
        plt.ylabel("Average CV", fontsize=12)
        plt.xlabel("Season", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(
            save_dir, "analysis_cv_by_season.png"), dpi=300)
        plt.close()

    # Plot 3: Judge Score vs. Estimated Fan Share (Scatter)
    if "judge_score" in estimates_df.columns:
        plt.figure(figsize=(10, 8))
        plot_df = estimates_df.dropna(
            subset=["judge_score", "est_fan_vote_mean"])

        sns.scatterplot(
            data=plot_df,
            x="judge_score",
            y="est_fan_vote_mean",
            alpha=0.3,
            color="#fc8d62",
            edgecolor=None
        )
        plt.title("Judge Scores vs. Estimated Fan Vote Share", fontsize=14)
        plt.xlabel("Total Judge Score (Raw)", fontsize=12)
        plt.ylabel("Estimated Fan Vote Share (Mean)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(
            save_dir, "analysis_judge_vs_fan.png"), dpi=300)
        plt.close()

    # Plot 4: Fan Vote Share Mean vs. Uncertainty (CV)
    if not estimates_df.empty:
        plt.figure(figsize=(10, 8))
        plot_df = estimates_df.dropna(subset=["cv", "est_fan_vote_mean"])
        sns.scatterplot(
            data=plot_df,
            x="est_fan_vote_mean",
            y="cv",
            alpha=0.3,
            color="#e78ac3",
            edgecolor=None
        )
        plt.title("Estimated Fan Vote Share vs. Uncertainty (CV)", fontsize=14)
        plt.xlabel("Estimated Vote Share (Mean)", fontsize=12)
        plt.ylabel("Uncertainty (CV)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "analysis_vote_vs_cv.png"), dpi=300)
        plt.close()

    print("Plots saved to:", save_dir)


# ==========================================
# Main
# ==========================================

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "Cleaned_data.csv")
    out_estimates = os.path.join(base_dir, "task1_estimates_complete.csv")
    out_week = os.path.join(base_dir, "task1_week_consistency.csv")

    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Simulation Configuration
    # fan_prior options: "dirichlet_uniform", "dirichlet_concentrated", "lognormal"
    estimates_df, week_df = estimate_fan_votes(
        df=df,
        n_samples=1000,           # Increase for better precision, e.g., 50000
        alpha_scale=1,           # Used if prior is dirichlet_concentrated
        lognormal_sigma=1,       # Used if prior is lognormal
        seed=42,
        fan_prior="dirichlet_uniform",
    )

    # Save Results
    estimates_df.to_csv(out_estimates, index=False)
    week_df.to_csv(out_week, index=False)
    print(f"\nSaved estimates to: {out_estimates}")
    print(f"Saved week summary to: {out_week}")

    # Print Summary Metrics
    if not week_df.empty:
        repro_rate = week_df["reproduced"].mean()
        pred_rate = week_df["pred_success"].mean()
        avg_cv = estimates_df["cv"].mean()

        print("\n=== Summary Metrics ===")
        print(f"Overall Consistency (Feasible Space Found): {repro_rate:.2%}")
        print(f"Prediction Success (Mean fits outcome):     {pred_rate:.2%}")
        print(f"Average Uncertainty (CV):                   {avg_cv:.3f}")

    # Generate Plots
    plot_analysis(estimates_df, week_df, base_dir)


if __name__ == "__main__":
    main()
