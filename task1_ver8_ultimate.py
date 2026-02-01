"""
Task 1 Ultimate: Performance-Popularity Mixture Model
Combines Global Solver and Final Inference into one unified workflow.

Core Logic:
1. Model Assumption: Fan Vote Share is a mixture of "Base Popularity" and "Weekly Performance".
   Expected_Share = (1 - LAMBDA) * Base_Popularity + LAMBDA * Normalized_Judge_Score
2. Global Solver: Iteratively estimates Base Popularity by "de-mixing" the valid samples from elimination constraints.
3. Final Inference: Uses the solved Base Popularity + Weekly Judge Scores to estimate posterior vote shares.
4. Constraints:
   - Regular Weeks: Elimination logic (Low score -> Eliminated).
   - Final Weeks: Strict placement logic (1st Place > 2nd Place in total correlation).

"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ==================== Config ====================
LAMBDA_PERF = 0.2       # Weight of Judge Performance in Fan Voting (0.0 - 1.0)
N_EPOCHS = 10           # Solver iterations
SAMPLES_PER_EPOCH = 5000
N_INFERENCE_SAMPLES = 20000
PRIOR_STRENGTH = 50.0   # Dictates the variance of the Dirichlet distribution
DATA_FILE = "Cleaned_data.csv"
OUTPUT_POPULARITY = "task1_ultimate_popularity.csv"
OUTPUT_ESTIMATES = "task1_ultimate_estimates.csv"

# ==================== Helpers ====================


def load_data(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, DATA_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
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
    # Handles ties by averaging, but for stability usually random or first is fine
    # Here we use simple argsort inversion
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
            combined = jr + fr  # Lower is better
            # Check if predicted order (ascending metric) matches placement order
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
        # The sum of ranks is maximized for the eliminated person?
        # Wait, Rank 1 is best. High Rank number is bad.
        # So "Lowest Combined Score" (Value) is Eliminated?
        # No, "Lowest Combined Score" (e.g. 1+1=2) is the WINNER.
        # "Highest Combined Score" (e.g. 10+10=20) is the LOSER.
        # So argmax() finds the worst person.
        return elim[(jr + fr).argmax()]
    elif method == "percent":
        jp = judge / judge.sum()
        # Sum of %: argmin is the lowest total score (Eliminated)
        return elim[(jp + fan).argmin()]
    elif method == "rank_judges_save":
        jr = compute_rank(judge)
        fr = compute_rank(fan)
        # Bottom 2 are candidates for elimination.
        # Check if true elimination is in Bottom 2 (indices of 2 largest sums)
        return elim[(jr + fr).argsort()[::-1][:2]].any()
    return False

# ==================== Part 1: Global Solver ====================


def solve_season_popularity(season_df: pd.DataFrame, season_num: int) -> Tuple[List[str], np.ndarray]:
    """
    Solves for Base Popularity (alpha) accounting for Judge Influence.
    """
    all_candidates = season_df["celebrity_name"].unique()
    n_candidates = len(all_candidates)
    name_to_idx = {name: i for i, name in enumerate(all_candidates)}

    # Initialize uniform Base Popularity
    # Note: These acts as relative weights, normalization happens at step level
    current_base_pop = np.ones(n_candidates) / n_candidates

    # Pre-process week data
    weeks_data = []
    max_week = season_df['week'].max()

    for week, g in season_df.groupby("week"):
        active_names = g["celebrity_name"].tolist()
        active_indices = [name_to_idx[n] for n in active_names]

        judge_scores = g["total_judge_score"].to_numpy(dtype=float)
        # Normalize judge scores to get "Performance Share"
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

    # Optimization Loop
    rng = np.random.default_rng(42)

    print(
        f"  > Solver: Processing Season {season_num} ({len(weeks_data)} valid weeks)...")

    for epoch in range(N_EPOCHS):
        accumulated_base = np.zeros(n_candidates)
        accumulated_counts = np.zeros(n_candidates)

        for wd in weeks_data:
            act_idx = wd["active_indices"]

            # 1. Construct Mixture Prior
            # Extract current base pop for active candidates
            local_base = current_base_pop[act_idx]
            # Normalize local base to sum to 1
            if local_base.sum() > 0:
                local_base = local_base / local_base.sum()
            else:
                local_base = np.ones(len(act_idx)) / len(act_idx)

            local_perf = wd["judge_share"]

            # MIXTURE FORMULA
            expected_share = (1 - LAMBDA_PERF) * \
                local_base + LAMBDA_PERF * local_perf

            # Dirichilet Alpha
            alpha_vec = expected_share * PRIOR_STRENGTH

            # 2. Sample
            samples = rng.dirichlet(alpha_vec, size=SAMPLES_PER_EPOCH)

            # 3. Filter
            valid_rows = []
            for i in range(SAMPLES_PER_EPOCH):
                if is_consistent(wd["judge_scores"], samples[i], wd["elim_mask"], wd["method"], wd.get("placements")):
                    valid_rows.append(samples[i])

            # 4. Backward Update (De-mixing)
            if valid_rows:
                valid_rows = np.array(valid_rows)
                # Average Valid Fan Share (Posterior Mean)
                posterior_mean = valid_rows.mean(axis=0)

                # INVERSE MIXTURE: Recover implied base popularity
                # Posterior ~ (1-L)*Base + L*Perf
                # Base ~ (Posterior - L*Perf) / (1-L)
                implied_base = (posterior_mean - LAMBDA_PERF *
                                local_perf) / (1 - LAMBDA_PERF)

                # Clip to prevent negative popularity (which is impossible)
                implied_base = np.maximum(1e-4, implied_base)

                # Accumulate back to global vector
                for i, global_idx in enumerate(act_idx):
                    accumulated_base[global_idx] += implied_base[i]
                    accumulated_counts[global_idx] += 1

        # End of Epoch: Update Global Vector
        mask = accumulated_counts > 0
        new_base = current_base_pop.copy()
        # Average the implied base values across all weeks
        new_base[mask] = accumulated_base[mask] / accumulated_counts[mask]

        # Soft update / Normalize
        new_base = new_base / new_base.sum()
        current_base_pop = new_base

    return all_candidates, current_base_pop

# ==================== Part 2: Final Inference ====================


def infer_season(season_df: pd.DataFrame, season_num: int, base_pop_map: Dict[str, float]) -> Tuple[List[Dict], int, int]:
    """
    Generates final stats and calculates prediction accuracy.
    Returns: (results_list, correct_count, total_count)
    """
    results = []
    rng = np.random.default_rng(2026)

    max_week = season_df['week'].max()
    method = method_for_season(season_num)

    # Track accuracy stats for this season
    season_correct = 0
    season_total = 0

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

        # 1. Mixture Prior
        local_base = np.array([base_pop_map.get(n, 0.01) for n in names])
        if local_base.sum() > 0:
            local_base /= local_base.sum()
        else:
            local_base = np.ones(len(names)) / len(names)

        expected_share = (1 - LAMBDA_PERF) * local_base + \
            LAMBDA_PERF * judge_share
        alpha_vec = expected_share * PRIOR_STRENGTH

        # 2. Sample
        samples = rng.dirichlet(alpha_vec, size=N_INFERENCE_SAMPLES)

        # 3. Filter
        feasible = []
        for i in range(N_INFERENCE_SAMPLES):
            if is_consistent(judge_scores, samples[i], elim_mask, method, placements if is_final_week else None):
                feasible.append(samples[i])

        # 4. Stats & Accuracy Check
        final_prediction = None

        if feasible:
            feasible = np.vstack(feasible)
            means = feasible.mean(axis=0)
            stds = feasible.std(axis=0)

            # Use the mean of valid samples as our "Final Prediction"
            final_prediction = means

            # 95% CI
            lower = np.percentile(feasible, 2.5, axis=0)
            upper = np.percentile(feasible, 97.5, axis=0)
            ci_widths = upper - lower

            for i, name in enumerate(names):
                results.append({
                    "season": season_num,
                    "week": week,
                    "celebrity_name": name,
                    "vote_share_est": means[i],
                    "std_dev": stds[i],
                    "ci_width_95": ci_widths[i],
                    "judge_score": judge_scores[i],
                    "base_popularity": local_base[i],
                    "perf_share": judge_share[i],
                    "is_eliminated": elim_mask[i]
                })
        else:
            # Fallback: Use expected share if no samples matched constraints
            final_prediction = expected_share

            for i, name in enumerate(names):
                results.append({
                    "season": season_num,
                    "week": week,
                    "celebrity_name": name,
                    "vote_share_est": expected_share[i],
                    "std_dev": np.nan,
                    "ci_width_95": np.nan,
                    "judge_score": judge_scores[i],
                    "base_popularity": local_base[i],
                    "perf_share": judge_share[i],
                    "is_eliminated": elim_mask[i]
                })

        # [NEW] Check Accuracy:
        # Does our calculated "Final Prediction" actually lead to the correct elimination/ranking?
        if is_consistent(judge_scores, final_prediction, elim_mask, method, placements if is_final_week else None):
            season_correct += 1
        season_total += 1

    return results, season_correct, season_total

# ==================== Main Execution ====================


def main():
    base_dir = os.path.dirname(__file__)
    try:
        df = load_data(base_dir)
    except Exception as e:
        print(e)
        return

    all_popularity = []
    all_estimates = []

    # Global Accuracy Counters
    global_correct_predictions = 0
    global_total_predictions = 0

    seasons = sorted(df["season"].unique())
    print(f"Starting Ultimate Pipeline for {len(seasons)} seasons...")
    print(f"Config: Lambda(Perf)={LAMBDA_PERF}, Epochs={N_EPOCHS}")

    for season in seasons:
        season_df = df[df["season"] == season]

        # Step 1: Solve Global Popularity (Iterative)
        candidates, base_shares = solve_season_popularity(season_df, season)

        # Store popularity
        pop_map = {}
        for n, s in zip(candidates, base_shares):
            all_popularity.append({
                "season": season,
                "celebrity_name": n,
                "base_popularity_share": s
            })
            pop_map[n] = s

        # Step 2: Final Inference (Sampling)
        # Unpack the 3 return values
        estimates, start_corr, s_tot = infer_season(season_df, season, pop_map)

        all_estimates.extend(estimates)
        global_correct_predictions += start_corr
        global_total_predictions += s_tot

    # Save Results
    pd.DataFrame(all_popularity).to_csv(
        os.path.join(base_dir, OUTPUT_POPULARITY), index=False)
    estimate_df = pd.DataFrame(all_estimates)
    estimate_df.to_csv(os.path.join(base_dir, OUTPUT_ESTIMATES), index=False)

    print(f"\nDone! Files saved:")
    print(f"1. {OUTPUT_POPULARITY} (Base Popularity)")
    print(f"2. {OUTPUT_ESTIMATES} (Final Vote Estimates)")

    # Print Accuracy Summary
    if global_total_predictions > 0:
        acc_pct = (global_correct_predictions / global_total_predictions) * 100
        print(f"\n{'='*40}")
        print(f"MODEL ACCURACY REPORT")
        print(f"{'='*40}")
        print(f"Total Weeks Evaluated: {global_total_predictions}")
        print(f"Correct Predictions:   {global_correct_predictions}")
        print(f"Overall Accuracy:      {acc_pct:.2f}%")
        print(f"{'='*40}\n")

    # Simple Visual Check
    if not estimate_df.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=estimate_df, x="judge_score",
                        y="vote_share_est", alpha=0.2, hue="is_eliminated")
        plt.title(
            f"Judge Score combined with Fan Vote (Lambda={LAMBDA_PERF})")
        plt.savefig(os.path.join(base_dir, "ultimate_correlation.png"))
        print("Saved ultimate_correlation.png")


if __name__ == "__main__":
    main()
