"""
Task 1: Estimate fan vote shares with Bayesian Time-Series Update.
Version 6: Uses posterior of week T-1 as prior for week T.

Key Features:
- Prior Propagation: Last week's feasible mean becomes this week's prior alpha center.
- Dynamic Alpha: Automatically handles eliminations by re-normalizing previous means.
- Prior Strength: Controls how strongly we stick to historical trends.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
    if method == "rank":
        judge_rank = compute_judge_rank(judge_scores)
        fan_rank = compute_fan_rank(fan_share)
        combined = judge_rank + fan_rank
        # highest total rank index (worst)
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
    rng: np.random.Generator | None = None,
    custom_alpha: np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Simulate a single week with a specific prior distribution.
    If custom_alpha is provided, uses Dirichlet(custom_alpha).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(judge_scores)

    # 如果没有提供先验，使用均匀分布 (alpha=1)
    if custom_alpha is None:
        alpha = np.ones(n)
    else:
        alpha = custom_alpha

    feasible = []
    total_feasible = 0

    # 简单的分批采样（Batch Sampling）优化性能
    # 每次生成 1000 个，循环直到达到 n_samples
    # 这里为了代码结构清晰，仍保持逻辑上的简单循环或一次性生成
    # 考虑到 n_samples=10000 并不大，直接一次性生成通常也没问题

    # Generate all potential fan shares at once
    # shape: (n_samples, n_contestants)
    candidates = rng.dirichlet(alpha, size=n_samples)

    for fan_share in candidates:
        if is_elimination_consistent(judge_scores, fan_share, eliminated_mask, method):
            feasible.append(fan_share)
            total_feasible += 1

    if feasible:
        return np.vstack(feasible), total_feasible
    return np.empty((0, n)), total_feasible


def estimate_fan_votes_with_memory(
    df: pd.DataFrame,
    n_samples: int = 20000,
    prior_strength: float = 30.0,  # 控制历史数据的权重
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)
    estimates = []
    week_results: List[WeekResult] = []

    # 必须保证按时间顺序处理
    groups = df.groupby(["season", "week"], sort=True)
    total_groups = len(groups)
    print(f"Processing {total_groups} weeks with time-series memory...")

    # 记忆体：{celebrity_name: last_mean_share}
    current_season = -1
    prev_means: Dict[str, float] = {}

    for (season, week), g in groups:
        season = int(season)
        week = int(week)

        # 如果进入新赛季，重置记忆
        if season != current_season:
            current_season = season
            prev_means = {}
            # print(f"--- Starting Season {season} ---")

        g = g.copy()
        method = method_for_season(season)

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        eliminated_mask = g["is_eliminated"].to_numpy(dtype=int).astype(bool)
        names = g["celebrity_name"].tolist()

        if eliminated_mask.sum() == 0:
            continue

        # ===== 构建动态先验 (Core Logic) =====
        n_contestants = len(names)

        # 1. 尝试从上一周获取均值
        prior_means = []
        for name in names:
            if name in prev_means:
                prior_means.append(prev_means[name])
            else:
                # 如果是该赛季第一周，或者某些特殊情况没记录，使用平均值
                prior_means.append(1.0 / n_contestants)  # 暂时用均匀基准，稍后归一化

        prior_means = np.array(prior_means)

        # 2. 如果之前记忆的总和不是1（这一定发生，因为有人被淘汰了），需要重新归一化
        #    这样保证了剩下的人按比例瓜分了被淘汰者的份额（Implicit Assumption）
        if prior_means.sum() > 0:
            prior_means = prior_means / prior_means.sum()
        else:
            prior_means = np.ones(n_contestants) / n_contestants

        # 3. 计算 Dirichlet 参数 Alpha
        #    Alpha = Mean * Strength
        #    第一周时，strength 可以设小一点吗？这里为了统一，加上一个基数平滑
        #    如果 prior_strength 很大，分布会非常集中在 prior_means 周围
        if not prev_means:
            # 赛季第一周，使用弱信息均匀分布 (alpha=1)
            custom_alpha = np.ones(n_contestants)
        else:
            # 后续周次，使用强信息先验
            # 加 0.1 避免 alpha 为 0
            custom_alpha = prior_means * prior_strength + 0.1

        # ===== 模拟 =====
        feasible_votes, total_feasible = simulate_week(
            judge_scores=judge_scores,
            eliminated_mask=eliminated_mask,
            method=method,
            n_samples=n_samples,
            rng=rng,
            custom_alpha=custom_alpha,
        )

        feasible_rate = total_feasible / n_samples
        reproduced = total_feasible > 0

        if feasible_votes.size == 0:
            week_results.append(WeekResult(
                season, week, method, feasible_rate, reproduced, False))
            # 预测失败，无法更新有意义的均值，但这周的人下周不在了也无所谓
            # 对于还在的人，或许应该保持上一周的均值？这里简单起见，不更新 prev_means 对于 failures
            for i, name in enumerate(names):
                estimates.append({
                    "season": season,
                    "week": week,
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

            # 更新记忆：供下周使用
            for name, m_val in zip(names, mean):
                prev_means[name] = float(m_val)

            pred_success = is_elimination_consistent(
                judge_scores, mean, eliminated_mask, method
            )
            week_results.append(WeekResult(
                season, week, method, feasible_rate, reproduced, bool(pred_success)))

            for i, (name, m, md, s) in enumerate(zip(names, mean, median, std)):
                cv_val = s / m if m > 1e-9 else np.nan
                estimates.append({
                    "season": season,
                    "week": week,
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

    # 1. CV Trend (Bar Plot with Zones)
    if not estimates_df.empty:
        season_cv = estimates_df.groupby("season")["cv"].mean().reset_index()
        plt.figure(figsize=(12, 6))

        # Draw bars
        sns.barplot(data=season_cv, x="season", y="cv",
                    color="#5DADE2", edgecolor="white")

        # Zones & Lines
        plt.ylim(0, 1.0)
        plt.axhline(y=0.1, color='#2ECC71', linestyle='--',
                    linewidth=1.5, label='High Certainty')
        plt.axhline(y=0.3, color='#E74C3C', linestyle='--',
                    linewidth=1.5, label='Low Certainty')
        plt.axhspan(0, 0.1, color='#2ECC71', alpha=0.1)
        plt.axhspan(0.1, 0.3, color='#F1C40F', alpha=0.05)
        plt.axhspan(0.3, 1.0, color='#E74C3C', alpha=0.05)

        plt.title(
            "Average Fan Vote Uncertainty (CV) by Season\n(With Time-Series Constraint)", fontsize=14)
        plt.ylabel("Average CV", fontsize=12)
        plt.xlabel("Season", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(
            save_dir, "analysis_cv_by_season_ts.png"), dpi=300)
        plt.close()

    print("Plots saved to:", save_dir)


# ==========================================
# Main
# ==========================================

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "Cleaned_data.csv")
    out_estimates = os.path.join(base_dir, "task1_estimates_ts.csv")
    out_week = os.path.join(base_dir, "task1_consistency_ts.csv")

    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Simulation Configuration
    # Prior Strength: 越高，历史依赖性越强，CV通常会越低（确定性越高）
    estimates_df, week_df = estimate_fan_votes_with_memory(
        df=df,
        n_samples=20000,
        prior_strength=40.0,  # 设定强度为 40，表示相当可信的历史惯性
        seed=42
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

        print("\n=== Summary Metrics (Time-Series Model) ===")
        print(f"Overall Consistency (Feasible Space Found): {repro_rate:.2%}")
        print(f"Prediction Success (Mean fits outcome):     {pred_rate:.2%}")
        print(f"Average Uncertainty (CV):                   {avg_cv:.3f}")
        print("Note: CV should be lower now due to prior concentration.")

    # Generate Plots
    plot_analysis(estimates_df, week_df, base_dir)


if __name__ == "__main__":
    main()
