"""
Task 1: Estimate fan vote shares using Season-Global Iterative Refinement.
Version 7: Treats the entire season as a single constraint problem to find
consistent 'Base Popularity' for each celebrity.

Core Logic:
1. Initialize celebrity alphas uniformly.
2. Iterate multiple rounds (epochs):
   a. For each week in season, sample votes using current alphas.
   b. Keep only samples that satisfy THAT week's elimination result.
   c. Aggregating ALL feasible samples from ALL weeks to update alphas.
3. This finds a "Base Popularity" vector that is globally consistent.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ==================== Config ====================
N_EPOCHS = 10           # 迭代轮数，模拟收敛过程
SAMPLES_PER_EPOCH = 5000  # 每轮每场采样的次数
PRIOR_STRENGTH = 50.0   # 控制收敛后分布的集中程度
OUTPUT_CSV = "task1_estimates_global.csv"
OUTPUT_PLOT_DIR = os.path.dirname(__file__)

# ==================== Helpers ====================


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["total_judge_score"] = pd.to_numeric(
        df["total_judge_score"], errors="coerce")
    return df[df["total_judge_score"].notna()].copy()


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
        # Final Week Logic: Rank Consistency Check
        # We enforce that better placement corresponds to better performance metric.
        if method == "rank":
            jr = compute_rank(judge)
            fr = compute_rank(fan)
            combined_metric = jr + fr  # Lower is better (Rank Sum)

            # Check correctness:
            # Sort indices by combined_metric (ascending -> best to worst)
            # The corresponding placements should be ascending (1, 2, 3...)
            # We relax strict inequality to allow ties, but generally, 1st place should beat 2nd.
            # However, Rejection Sampling is strict. Let's try strict monotonicity for Placement 1 vs others.

            # Simple pairwise check: for every pair, if Placement[i] < Placement[j], then Metric[i] <= Metric[j].
            # Actually, let's use a simpler proxy: The predicted order matches the placement order.

            pred_order_indices = np.argsort(
                combined_metric)  # Indices of Best -> Worst
            pred_placements = placement[pred_order_indices]

            # Check if pred_placements is sorted (ascending)
            return np.all(pred_placements[:-1] <= pred_placements[1:])

        elif method == "percent":
            jp = judge / judge.sum()
            combined_metric = jp + fan  # Higher is better

            # Sort indices by combined_metric (descending -> best to worst)
            pred_order_indices = np.argsort(combined_metric)[::-1]
            pred_placements = placement[pred_order_indices]

            return np.all(pred_placements[:-1] <= pred_placements[1:])

        elif method == "rank_judges_save":
            # Similar to rank
            jr = compute_rank(judge)
            fr = compute_rank(fan)
            combined_metric = jr + fr
            pred_order_indices = np.argsort(combined_metric)
            pred_placements = placement[pred_order_indices]
            return np.all(pred_placements[:-1] <= pred_placements[1:])

    # Regular Week Logic
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
        # Bottom 2 checks
        return elim[(jr + fr).argsort()[::-1][:2]].any()
    return False

# ==================== Global Solver ====================


def solve_season_popularity(season_df: pd.DataFrame, season_num: int):
    """
    Solve for base popularity for a single season iteratively.
    """
    # 1. 准备数据结构
    all_candidates = season_df["celebrity_name"].unique()
    n_candidates = len(all_candidates)
    name_to_idx = {name: i for i, name in enumerate(all_candidates)}

    current_alpha = np.ones(n_candidates) * (PRIOR_STRENGTH / n_candidates)

    weeks_data = []
    max_week = season_df['week'].max()

    for week, g in season_df.groupby("week"):
        active_names = g["celebrity_name"].tolist()
        active_indices = [name_to_idx[n] for n in active_names]

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        elim_mask = g["is_eliminated"].to_numpy(dtype=int).astype(bool)

        # New: Placement data for final week check
        placements = g["placement"].to_numpy(dtype=float)

        method = method_for_season(season_num)

        is_final_week = (week == max_week)

        if elim_mask.sum() > 0 or is_final_week:
            weeks_data.append({
                "week": week,
                "active_indices": active_indices,
                "judge_scores": judge_scores,
                "elim_mask": elim_mask,
                "placements": placements if is_final_week else None,
                "method": method
            })

    # 2. 迭代优化
    print(
        f"Season {season_num}: Solving global popularity ({len(weeks_data)} active weeks)...")

    rng = np.random.default_rng(42)

    for epoch in range(N_EPOCHS):
        # 用于累积这一轮所有符合条件的样本，以此更新 alpha
        accumulated_shares = np.zeros(n_candidates)
        accumulated_counts = np.zeros(n_candidates)  # 记录每个人出现了多少次有效样本

        # 遍历每一周
        total_season_feasible = 0

        for wd in weeks_data:
            act_idx = wd["active_indices"]

            # 当前这些活跃选手的 Alpha
            # 注意：采样时只需考虑当前在场的人。
            # 关键：从全局 Alpha 中取出对应子集，并重新归一化，作为本周的先验
            sub_alpha = current_alpha[act_idx]

            # 生成样本 (Dirichlet)
            # shape: (SAMPLES, n_active)
            samples = rng.dirichlet(sub_alpha, size=SAMPLES_PER_EPOCH)

            # 验证一致性
            valid_rows = []
            for i in range(SAMPLES_PER_EPOCH):
                if is_consistent(wd["judge_scores"], samples[i], wd["elim_mask"], wd["method"], wd.get("placements")):
                    valid_rows.append(samples[i])

            # 收集有效样本
            if valid_rows:
                valid_rows = np.array(valid_rows)
                total_season_feasible += len(valid_rows)

                # 将有效份额累加回全局累加器
                # 这种做法假设：如果某人在第一周得票率是 0.1，到第五周剩下人少了，他的得票率自然会上升
                # 因此我们需要把份额“还原”到全局尺度吗？
                # 这是一个难点。如果不还原，第10周得票率天然比第1周高（分母小了）。
                # 更有力的逻辑：我们优化的是 "Relative Strength" (Alpha)。
                # Dirichlet的性质：如果 X ~ Dir(alpha1...alphaN)，那么去掉一项后的归一化分布就是 Dir(去掉那个alpha)。
                # 所以我们直接累加 valid_rows * weight 好像不太对。

                # 修正策略：
                # 我们不累加具体的 share 值，而是累加 "Implied Alpha Update"。
                # 简单近似：如果某人在本周平均拿了 20% 的票，且本周 Alpha sum = S，
                # 那么这暗示他的 Alpha 应该是 0.2 * S。
                current_sub_alpha_sum = sub_alpha.sum()
                implied_alphas = valid_rows.mean(
                    axis=0) * current_sub_alpha_sum

                # 更新累加器
                for local_i, global_i in enumerate(act_idx):
                    accumulated_shares[global_i] += implied_alphas[local_i]
                    accumulated_counts[global_i] += 1

        # End of Epoch: Update Global Alphas
        # 新的 Alpha = 平均 Implied Alpha
        # 避免除零
        mask = accumulated_counts > 0
        new_alphas = current_alpha.copy()
        new_alphas[mask] = accumulated_shares[mask] / accumulated_counts[mask]

        # 引入学习率或动量可以是改进点，这里直接替换
        current_alpha = new_alphas

        # 归一化以保持尺度 (Sum = PRIOR_STRENGTH)
        current_alpha = current_alpha / current_alpha.sum() * PRIOR_STRENGTH

        # print(f"  Epoch {epoch+1} done. Avg Feasible/Week: {total_season_feasible/len(weeks_data):.1f}")

    # 3. 最终输出
    # 收敛后的 current_alpha 即为这赛季各名人的 Base Popularity Score
    # 我们可以把它转化为第一周（全员在场）的期望得票率：
    base_shares = current_alpha / current_alpha.sum()

    return all_candidates, base_shares

# ==================== Main ====================


def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "Cleaned_data.csv")

    if not os.path.exists(data_path):
        print("Data not found.")
        return

    df = load_data(data_path)

    global_results = []

    # 按赛季处理
    for season in sorted(df["season"].unique()):
        season_df = df[df["season"] == season]
        names, shares = solve_season_popularity(season_df, season)

        for n, s in zip(names, shares):
            global_results.append({
                "season": season,
                "celebrity_name": n,
                "base_popularity_share": s
            })

    res_df = pd.DataFrame(global_results)
    metric_path = os.path.join(base_dir, "task1_global_popularity.csv")
    res_df.to_csv(metric_path, index=False)
    print(f"Saved global popularity metrics to {metric_path}")

    # 绘图：展示某几个赛季的人气分布
    # Heatmap of base popularity? Or Barplot for Season 2 (Controversial)

    # Example: Season 2
    s2 = res_df[res_df["season"] == 2].sort_values(
        "base_popularity_share", ascending=False)
    if not s2.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=s2, x="celebrity_name",
                    y="base_popularity_share", palette="viridis")
        plt.title(
            "Season 2: Global Base Popularity Estimates (Constraint Inversion)", fontsize=14)
        plt.xticks(rotation=45)
        plt.ylabel("Base Vote Share (Estimated)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "season_2_popularity.png"))
        print("Saved season_2_popularity.png")


if __name__ == "__main__":
    main()
