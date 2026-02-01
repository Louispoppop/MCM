"""
Task 1: Estimate fan vote shares using rejection-sampling Monte Carlo.

Inputs:
    - Cleaned_data.csv (season, week, celebrity_name, total_judge_score, is_eliminated, ...)

Outputs:
    - task1_fan_vote_estimates.csv
      Columns: season, week, celebrity_name, est_fan_vote_mean, est_fan_vote_median,
               est_fan_vote_std, feasible_samples
    - task1_week_consistency.csv
      Columns: season, week, method, feasible_rate, reproduced

Notes:
    - Seasons 1-2: rank-based combination
    - Seasons 3-27: percent-based combination
    - Seasons 28+: rank-based + judges' save (bottom-two check)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class WeekResult:
    season: int
    week: int
    method: str
    feasible_rate: float
    reproduced: bool
    pred_success: bool


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only rows that are active in this week (nonzero scores imply active in the cleaned data)
    # If needed, adjust this filter based on your dataset conventions.
    df = df.copy()
    df["total_judge_score"] = pd.to_numeric(
        df["total_judge_score"], errors="coerce")
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
    # Rank with average method for ties
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
        eliminated_pred = combined.argmax()  # highest rank sum => worst
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


def simulate_week(
    judge_scores: np.ndarray,
    eliminated_mask: np.ndarray,
    method: str,
    n_samples: int = 10000,
    alpha_scale: float = 0.3,
    rng: np.random.Generator | None = None,
    fan_prior: str = "lognormal",  # 新增：选择分布
) -> Tuple[np.ndarray, int]:
    """
    Returns:
        feasible_votes: ndarray (k, n_contestants)
        total_feasible: int
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(judge_scores)

    # ===== Fan vote prior options (independent of judge scores) =====
    def sample_fan_share() -> np.ndarray:
        # Option A (default): Uniform Dirichlet (no relation to judges)
        if fan_prior == "dirichlet_uniform":
            return rng.dirichlet(np.ones(n))

        # Option B: Dirichlet with mild concentration (still independent)
        if fan_prior == "dirichlet_concentrated":
            return rng.dirichlet(np.ones(n) * alpha_scale)

        # Option C: Lognormal + normalize (heavier tails)
        if fan_prior == "lognormal":
            x = rng.lognormal(mean=0.0, sigma=2, size=n)
            return x / x.sum()

        # Option D: Normal -> softmax (symmetric)
        if fan_prior == "normal_softmax":
            x = rng.normal(loc=0.0, scale=1.0, size=n)
            e = np.exp(x - x.max())
            return e / e.sum()

        raise ValueError(f"Unknown fan_prior: {fan_prior}")

    feasible = []
    total_feasible = 0

    for _ in range(n_samples):
        fan_share = sample_fan_share()

        if method == "rank":
            judge_rank = compute_judge_rank(judge_scores)
            fan_rank = compute_fan_rank(fan_share)
            combined = judge_rank + fan_rank
            eliminated_pred = combined.argmax()  # highest rank sum => worst
            if eliminated_mask[eliminated_pred]:
                feasible.append(fan_share)
                total_feasible += 1

        elif method == "percent":
            judge_percent = judge_scores / judge_scores.sum()
            combined = judge_percent + fan_share
            eliminated_pred = combined.argmin()
            if eliminated_mask[eliminated_pred]:
                feasible.append(fan_share)
                total_feasible += 1

        elif method == "rank_judges_save":
            # bottom two by rank sum; true eliminated must be in bottom two
            judge_rank = compute_judge_rank(judge_scores)
            fan_rank = compute_fan_rank(fan_share)
            combined = judge_rank + fan_rank
            bottom_two = combined.argsort()[::-1][:2]
            if eliminated_mask[bottom_two].any():
                feasible.append(fan_share)
                total_feasible += 1
        else:
            raise ValueError(f"Unknown method: {method}")

    if feasible:
        return np.vstack(feasible), total_feasible
    return np.empty((0, n)), total_feasible


def estimate_fan_votes(
    df: pd.DataFrame,
    n_samples: int = 10000,
    alpha_scale: float = 8.0,
    seed: int = 42,
    fan_prior: str = "lognormal",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    estimates = []
    week_results: List[WeekResult] = []

    for (season, week), g in df.groupby(["season", "week"], sort=True):
        g = g.copy()
        method = method_for_season(int(season))

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        eliminated_mask = g["is_eliminated"].to_numpy(dtype=int).astype(bool)

        # Skip weeks with no elimination info
        if eliminated_mask.sum() == 0:
            continue

        feasible_votes, total_feasible = simulate_week(
            judge_scores=judge_scores,
            eliminated_mask=eliminated_mask,
            method=method,
            n_samples=n_samples,
            alpha_scale=alpha_scale,
            rng=rng,
            fan_prior=fan_prior,
        )

        feasible_rate = total_feasible / n_samples
        reproduced = total_feasible > 0  # 等价于可行集非空，即模型找到了符合实际淘汰结果的解

        if feasible_votes.size == 0:
            week_results.append(
                WeekResult(
                    int(season),
                    int(week),
                    method,
                    feasible_rate,
                    reproduced,
                    pred_success=False,
                )
            )
            # If no feasible samples found, fall back to NaN estimates
            for name in g["celebrity_name"].tolist():
                estimates.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "celebrity_name": name,
                        "est_fan_vote_mean": np.nan,
                        "est_fan_vote_median": np.nan,
                        "est_fan_vote_std": np.nan,
                        "feasible_samples": 0,
                    }
                )
            continue

        mean = feasible_votes.mean(axis=0)
        median = np.median(feasible_votes, axis=0)
        std = feasible_votes.std(axis=0)

        pred_success = is_elimination_consistent(
            judge_scores=judge_scores,
            fan_share=mean,
            eliminated_mask=eliminated_mask,
            method=method,
        )
        week_results.append(
            WeekResult(
                int(season),
                int(week),
                method,
                feasible_rate,
                reproduced,
                pred_success=bool(pred_success),
            )
        )

        for name, m, md, s in zip(g["celebrity_name"], mean, median, std):
            estimates.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "celebrity_name": name,
                    "est_fan_vote_mean": float(m),
                    "est_fan_vote_median": float(md),
                    "est_fan_vote_std": float(s),
                    "feasible_samples": int(total_feasible),
                }
            )

    estimates_df = pd.DataFrame(estimates)
    week_df = pd.DataFrame([wr.__dict__ for wr in week_results])
    return estimates_df, week_df


def main() -> None:
    data_path = os.path.join(os.path.dirname(__file__), "Cleaned_data.csv")
    out_estimates = os.path.join(os.path.dirname(
        __file__), "task1_estimates.csv")
    out_week = os.path.join(os.path.dirname(__file__),
                            "task1_consistency.csv")

    df = load_data(data_path)

    estimates_df, week_df = estimate_fan_votes(
        df=df,
        n_samples=20000,
        seed=42,
        fan_prior="dirichlet_concentrated",
    )

    estimates_df.to_csv(out_estimates, index=False)
    week_df.to_csv(out_week, index=False)

    print("Saved:")
    print(out_estimates)
    print(out_week)

    if not week_df.empty:
        # 一致性：模型能够复现实际淘汰结果的周数比例
        consistency = week_df["reproduced"].mean()
        print(f"Consistency (Overall Reproduced Rate): {consistency:.3f}")
        pred_success_rate = week_df["pred_success"].mean()
        print(
            f"Prediction Success Rate (Mean-in-Feasible): {pred_success_rate:.3f}")
        avg_feasible_rate = week_df["feasible_rate"].mean()
        print(f"Average Feasible Sample Rate: {avg_feasible_rate:.5f}")


if __name__ == "__main__":
    main()
