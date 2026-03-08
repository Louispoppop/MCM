"""
Model 1 Sensitivity Analysis
Analyzes the impact of Lambda (Performance Weight) and Kappa (Prior Strength/Variance)
on the model's ability to reproduce historical elimination results.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import itertools
from tqdm import tqdm

# ==================== Config for Sensitivity Analysis ====================
# We will iterate over these ranges
LAMBDA_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
KAPPA_RANGE = [10, 30, 50, 70, 90]

# Reduced sampling for speed during sensitivity analysis loop
# (Original: Epochs=10, SolverSamples=5000, InfSamples=20000)
SENS_EPOCHS = 5
SENS_SOLVER_SAMPLES = 2000
SENS_INF_SAMPLES = 5000

DATA_FILE = "Cleaned_data.csv"
OUTPUT_CSV = "sensitivity_results.csv"
OUTPUT_PLOT = "sensitivity_heatmap.png"

# ==================== Helpers (Copied & Adapted) ====================


def load_data(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, DATA_FILE)
    if not os.path.exists(path):
        # Fallback to checking parent dir if running from subfolder
        path = os.path.join(os.path.dirname(base_dir), DATA_FILE)
        if not os.path.exists(path):
            # Try absolute path based on workspace info if relative fails
            path = r"c:\Users\Xiangkun\MCM\Cleaned_data.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{DATA_FILE} not found in {base_dir} or parent.")

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
    # 1. Final Week Constraint
    if placement is not None:
        if method == "rank":
            jr = compute_rank(judge)
            fr = compute_rank(fan)
            combined = jr + fr  # Lower is better (Rank 1 + Rank 1 = 2)
            pred_order = np.argsort(combined)
            pred_placements = placement[pred_order]
            return np.all(pred_placements[:-1] <= pred_placements[1:])

        elif method == "percent":
            jp = judge / judge.sum()
            combined = jp + fan  # Higher is better
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

    # 2. Regular Elimination Constraint
    if method == "rank":
        jr = compute_rank(judge)
        fr = compute_rank(fan)
        # Max of Sum(Rank) is eliminated (Rank 1 is best, Rank N is worst)
        return elim[(jr + fr).argmax()]
    elif method == "percent":
        jp = judge / judge.sum()
        # Min of Sum(Score) is eliminated
        return elim[(jp + fan).argmin()]
    elif method == "rank_judges_save":
        jr = compute_rank(judge)
        fr = compute_rank(fan)
        return elim[(jr + fr).argsort()[::-1][:2]].any()
    return False

# ==================== Core Logic Parameterized ====================


def solve_season_popularity_param(season_df: pd.DataFrame, season_num: int,
                                  lambda_val: float, kappa_val: float) -> Tuple[List[str], np.ndarray]:
    """
    Solves for Base Popularity with dynamic parameters.
    """
    all_candidates = season_df["celebrity_name"].unique()
    n_candidates = len(all_candidates)
    name_to_idx = {name: i for i, name in enumerate(all_candidates)}

    current_base_pop = np.ones(n_candidates) / n_candidates

    weeks_data = []
    max_week = season_df['week'].max()

    for week, g in season_df.groupby("week"):
        active_names = g["celebrity_name"].tolist()
        active_indices = [name_to_idx[n] for n in active_names]

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        if judge_scores.sum() > 0:
            judge_share = judge_scores / judge_scores.sum()
        else:
            judge_share = np.ones_like(judge_scores) / len(judge_scores)

        elim_mask = g["is_eliminated"].to_numpy(dtype=int).astype(bool)
        placements = g["placement"].to_numpy(dtype=float)
        method = method_for_season(season_num)
        is_final_week = (week == max_week)

        if elim_mask.sum() > 0 or is_final_week:
            weeks_data.append({
                "active_indices": active_indices,
                "judge_scores": judge_scores,
                "judge_share": judge_share,
                "elim_mask": elim_mask,
                "placements": placements if is_final_week else None,
                "method": method
            })

    rng = np.random.default_rng(42)

    # Simplified Loop for Sensitivity
    for epoch in range(SENS_EPOCHS):
        accumulated_base = np.zeros(n_candidates)
        accumulated_counts = np.zeros(n_candidates)

        for wd in weeks_data:
            act_idx = wd["active_indices"]
            local_base = current_base_pop[act_idx]
            if local_base.sum() > 0:
                local_base = local_base / local_base.sum()
            else:
                local_base = np.ones(len(act_idx)) / len(act_idx)

            local_perf = wd["judge_share"]
            expected_share = (1 - lambda_val) * local_base + \
                lambda_val * local_perf
            alpha_vec = expected_share * kappa_val

            samples = rng.dirichlet(alpha_vec, size=SENS_SOLVER_SAMPLES)

            # Fast validation using vectorization where possible or just loop
            # Since check is complex, we loop
            valid_rows = []
            for i in range(SENS_SOLVER_SAMPLES):
                if is_consistent(wd["judge_scores"], samples[i], wd["elim_mask"], wd["method"], wd.get("placements")):
                    valid_rows.append(samples[i])

            if valid_rows:
                valid_rows = np.array(valid_rows)
                posterior_mean = valid_rows.mean(axis=0)

                # De-mixing
                # Base ~ (Posterior - L*Perf) / (1-L)
                # Avoid divide by zero if lambda is 1 (unlikely range)
                denom = 1 - lambda_val
                if denom < 1e-6:
                    denom = 1e-6

                implied_base = (posterior_mean - lambda_val *
                                local_perf) / denom
                implied_base = np.maximum(1e-4, implied_base)

                for i, global_idx in enumerate(act_idx):
                    accumulated_base[global_idx] += implied_base[i]
                    accumulated_counts[global_idx] += 1

        mask = accumulated_counts > 0
        new_base = current_base_pop.copy()
        new_base[mask] = accumulated_base[mask] / accumulated_counts[mask]
        new_base = new_base / new_base.sum()
        current_base_pop = new_base

    return all_candidates, current_base_pop


def infer_season_param(season_df: pd.DataFrame, season_num: int, base_pop_map: Dict[str, float],
                       lambda_val: float, kappa_val: float) -> Tuple[int, int, float, int]:
    """
    Returns (correct_count, total_count, sum_cv, count_cv) for the season
    """
    rng = np.random.default_rng(2026)
    max_week = season_df['week'].max()
    method = method_for_season(season_num)

    season_correct = 0
    season_total = 0
    season_cv_sum = 0.0
    season_cv_count = 0

    groups = season_df.groupby("week")
    for week, g in groups:
        is_final_week = (week == max_week)
        names = g["celebrity_name"].tolist()
        elim_mask = g["is_eliminated"].to_numpy(dtype=int).astype(bool)
        placements = g["placement"].to_numpy(dtype=float)

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        if judge_scores.sum() > 0:
            judge_share = judge_scores / judge_scores.sum()
        else:
            judge_share = np.ones_like(judge_scores) / len(judge_scores)

        if elim_mask.sum() == 0 and not is_final_week:
            continue

        local_base = np.array([base_pop_map.get(n, 0.01) for n in names])
        if local_base.sum() > 0:
            local_base /= local_base.sum()
        else:
            local_base = np.ones(len(names)) / len(names)

        expected_share = (1 - lambda_val) * local_base + \
            lambda_val * judge_share
        alpha_vec = expected_share * kappa_val

        samples = rng.dirichlet(alpha_vec, size=SENS_INF_SAMPLES)

        feasible = []
        for i in range(SENS_INF_SAMPLES):
            if is_consistent(judge_scores, samples[i], elim_mask, method, placements if is_final_week else None):
                feasible.append(samples[i])

        final_prediction = None
        if feasible:
            feasible = np.vstack(feasible)
            means = feasible.mean(axis=0)
            stds = feasible.std(axis=0)
            final_prediction = means

            # Calculate CV
            # Clip mean to avoiding division by zero
            safe_means = np.maximum(means, 1e-6)
            cvs = stds / safe_means
            season_cv_sum += cvs.sum()
            season_cv_count += len(cvs)
        else:
            final_prediction = expected_share

        if is_consistent(judge_scores, final_prediction, elim_mask, method, placements if is_final_week else None):
            season_correct += 1
        season_total += 1

    return season_correct, season_total, season_cv_sum, season_cv_count

# ==================== Main Analysis Loop ====================


def main():
    base_dir = os.path.dirname(__file__)
    print(f"Loading data from {base_dir}...")
    try:
        df = load_data(base_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # To save time, we can run on a representative subset of seasons if needed.
    # But for a paper, better to run all. ~34 seasons.
    # To make it faster, we reduced EPOCHS and SAMPLES constants above.
    seasons = sorted(df["season"].unique())
    # seasons = seasons[::5] # Uncomment to run on fewer seasons for debug (e.g. every 5th season)

    results_grid = []

    print(f"Starting Sensitivity Analysis...")
    print(f"Lambda Range: {LAMBDA_RANGE}")
    print(f"Kappa Range: {KAPPA_RANGE}")

    total_combinations = len(LAMBDA_RANGE) * len(KAPPA_RANGE)
    current_combo = 0

    for l_val in LAMBDA_RANGE:
        for k_val in KAPPA_RANGE:
            current_combo += 1
            print(
                f"[{current_combo}/{total_combinations}] Testing Lambda={l_val}, Kappa={k_val}...")

            combo_correct = 0
            combo_total = 0
            combo_cv_sum = 0.0
            combo_cv_count = 0

            # Run for all seasons with this parameter set
            for season in tqdm(seasons, leave=False):
                season_df = df[df["season"] == season]

                # 1. Solve
                candidates, base_shares = solve_season_popularity_param(
                    season_df, season, l_val, k_val)
                pop_map = {n: s for n, s in zip(candidates, base_shares)}

                # 2. Infer & Check Accuracy
                corr, tot, cv_s, cv_c = infer_season_param(
                    season_df, season, pop_map, l_val, k_val)
                combo_correct += corr
                combo_total += tot
                combo_cv_sum += cv_s
                combo_cv_count += cv_c

            accuracy = combo_correct / combo_total if combo_total > 0 else 0
            avg_cv = combo_cv_sum / combo_cv_count if combo_cv_count > 0 else 0
            print(f"   -> Accuracy: {accuracy:.4f}, Avg CV: {avg_cv:.4f}")

            results_grid.append({
                "Lambda": l_val,
                "Kappa": k_val,
                "Accuracy": accuracy,
                "AvgCV": avg_cv
            })

    # ==================== Saving & Plotting ====================
    results_df = pd.DataFrame(results_grid)
    results_df.to_csv(os.path.join(base_dir, OUTPUT_CSV), index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Plot 1: Accuracy Heatmap
    pivot_acc = results_df.pivot(
        index="Lambda", columns="Kappa", values="Accuracy")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="viridis",
                cbar_kws={'label': 'Historical Reproduction Rate'})
    plt.title("Model I Sensitivity: Accuracy vs Params")
    plt.xlabel("Kappa (Variance Control)")
    plt.ylabel("Lambda (Performance Weight)")

    plot_path = os.path.join(base_dir, OUTPUT_PLOT)
    plt.savefig(plot_path)
    print(f"Accuracy Heatmap saved to {plot_path}")

    # Plot 2: CV Heatmap
    pivot_cv = results_df.pivot(
        index="Lambda", columns="Kappa", values="AvgCV")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_cv, annot=True, fmt=".3f", cmap="magma_r",
                cbar_kws={'label': 'Average Coefficient of Variation (CV)'})
    plt.title("Model I Sensitivity: Uncertainty (CV) vs Params")
    plt.xlabel("Kappa (Variance Control)")
    plt.ylabel("Lambda (Performance Weight)")

    cv_plot_path = os.path.join(base_dir, "sensitivity_cv_heatmap.png")
    plt.savefig(cv_plot_path)
    print(f"CV Heatmap saved to {cv_plot_path}")


if __name__ == "__main__":
    main()
