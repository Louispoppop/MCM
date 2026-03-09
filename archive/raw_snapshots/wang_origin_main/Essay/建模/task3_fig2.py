"""Visualize how celebrity features affect judges vs fan responses.

This script compares the *regression coefficients* (not normalized)
for two targets:
- x-axis side: coefficient for avg_score_rate (judges)
- y-axis side: coefficient for avg_voting_rate (fans)

Several figures are generated:
- A scatter plot (with a nonlinear axis stretch) showing judges vs
    fans coefficients for each predictor, together with the x=y line.
- A rotated scatter plot using "average effect" vs "difference"
    between judges and fans.
- A dumbbell chart where each horizontal line is one predictor,
    connecting its judges vs fans coefficients.

All values use the same raw coefficients as in mixedlm_results.json
and in 建模.md, so numbers like popularity_baseline ≈ 0.676 (judges)
and ≈ 0.964 (fans) will appear at相应的位置，只在坐标上做非线性
拉伸以改善可视化效果。
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


BASE_DIR = Path(__file__).resolve().parent
RESULT_PATH = BASE_DIR / "outputs" / "mixedlm_results.json"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"


def main() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Cannot find mixedlm_results.json at {RESULT_PATH}")

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        models = json.load(f)

    score_params = models.get("avg_score_rate", {}).get("params", {})
    vote_params = models.get("avg_voting_rate", {}).get("params", {})

    # Remove random-effect variance keys if present
    score_params = {k: v for k, v in score_params.items() if not k.startswith("Group")}
    vote_params = {k: v for k, v in vote_params.items() if not k.startswith("Group")}

    predictors = sorted(set(score_params.keys()) & set(vote_params.keys()))
    if not predictors:
        raise ValueError(
            "No common predictors between avg_score_rate and avg_voting_rate."
        )

    rows = []
    for p in predictors:
        cs = float(score_params[p])
        cv = float(vote_params[p])
        rows.append(
            {
                "predictor": p,
                "coef_score": cs,
                "coef_vote": cv,
            }
        )

    df = pd.DataFrame(rows)

    # Use raw coefficients directly (no normalization), to match 建模.md
    df["coef_score_norm"] = df["coef_score"]
    df["coef_vote_norm"] = df["coef_vote"]
    df["diff"] = df["coef_score_norm"] - df["coef_vote_norm"]

    # Decide axis domain and custom stretch:
    # - Clamp the lower bound at -0.25 (we know there are no coefficients below),
    # - Redistribute visual length so that 0–0.25 is stretched compared with
    #   [-0.25, 0] and [0.25, max].
    all_coef = np.concatenate(
        [
            df["coef_score_norm"].to_numpy(),
            df["coef_vote_norm"].to_numpy(),
        ]
    )
    coef_min, coef_max = float(all_coef.min()), float(all_coef.max())
    lo_raw = -0.25
    hi_raw = coef_max + 0.02  # small padding on the positive side

    # Piecewise-linear stretch:
    # [-0.25, 0]  -> visual length 1
    # [0, 0.25]   -> visual length 3 (stretched)
    # [0.25, hi]  -> visual length 2
    def _stretch(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        # clip to designed domain
        arr = np.maximum(arr, lo_raw)
        b0, b1, b2, b3 = lo_raw, 0.0, 0.25, hi_raw
        len1, len2, len3 = 1.0, 3.0, 2.0
        s1 = len1 / (b1 - b0) if b1 > b0 else 0.0
        s2 = len2 / (b2 - b1) if b2 > b1 else 0.0
        s3 = len3 / (b3 - b2) if b3 > b2 else 0.0

        out = np.empty_like(arr)
        mask1 = arr <= b1
        mask2 = (arr > b1) & (arr <= b2)
        mask3 = arr > b2

        out[mask1] = s1 * (arr[mask1] - b0)
        out[mask2] = len1 + s2 * (arr[mask2] - b1)
        if np.any(mask3):
            if s3 > 0:
                out[mask3] = len1 + len2 + s3 * (arr[mask3] - b2)
            else:
                out[mask3] = len1 + len2
        return out

    df["coef_score_view"] = _stretch(df["coef_score_norm"].to_numpy())
    df["coef_vote_view"] = _stretch(df["coef_vote_norm"].to_numpy())

    # Build scatter plot (in stretched coordinates, but轴刻度显示原始系数)
    fig, ax = plt.subplots(figsize=(7.5, 7))

    # Color by difference (score - vote): warm = judges more sensitive, cool = fans more sensitive
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-abs(df["diff"]).max(), vmax=abs(df["diff"]).max())
    colors = cmap(norm(df["diff"].to_numpy()))

    ax.scatter(
        df["coef_score_view"],
        df["coef_vote_view"],
        s=80,
        c=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Diagonal reference line: equal effect on judges & fans (in stretched space)
    grid = np.linspace(lo_raw, hi_raw, 200)
    grid_stretched = _stretch(grid)
    ax.plot(
        grid_stretched, grid_stretched, color="#999999", linestyle="--", linewidth=1
    )

    ax.set_xlim(grid_stretched.min(), grid_stretched.max())
    ax.set_ylim(grid_stretched.min(), grid_stretched.max())

    # Label each point with the predictor name
    for _, row in df.iterrows():
        ax.text(
            row["coef_score_view"],
            row["coef_vote_view"],
            " " + row["predictor"],
            fontsize=8,
            ha="left",
            va="center",
        )

    # Grey reference lines at coefficient 0 (after stretching)
    zero_view = _stretch(np.array([0.0]))[0]
    ax.axhline(zero_view, color="#cccccc", linewidth=0.8)
    ax.axvline(zero_view, color="#cccccc", linewidth=0.8)

    # Non-uniform ticks: show original coefficient values but spaced via stretch
    # Build a fixed candidate set ensuring 0 is always included
    tick_candidates = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Keep ticks within the raw domain actually used
    tick_vals = tick_candidates[
        (tick_candidates >= lo_raw) & (tick_candidates <= hi_raw)
    ]
    tick_pos = _stretch(tick_vals)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{v:.2g}" for v in tick_vals])
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f"{v:.2g}" for v in tick_vals])

    ax.set_xlabel("Coefficient for avg_score_rate (judges)")
    ax.set_ylabel("Coefficient for avg_voting_rate (fans)")
    ax.set_title("Celebrity feature effects: judges vs fans (stretched axes)")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8)
    cbar.set_label("coef_score - coef_vote (judges minus fans)")

    out_path = OUT_DIR / "fig_coef_judges_vs_fans.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- Figure 2: rotated coordinates (average vs difference) ---
    # Transform to mean effect (along diagonal) and disagreement (perpendicular)
    df["mean_norm"] = 0.5 * (df["coef_score_norm"] + df["coef_vote_norm"])
    df["diff_norm"] = 0.5 * (df["coef_score_norm"] - df["coef_vote_norm"])

    fig2, ax2 = plt.subplots(figsize=(7.5, 7))
    ax2.scatter(
        df["mean_norm"],
        df["diff_norm"],
        s=80,
        c=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Axes and limits
    x_all = df["mean_norm"].to_numpy()
    y_all = df["diff_norm"].to_numpy()
    x_lim = max(float(np.abs(x_all).max()), 0.1)
    y_lim = max(float(np.abs(y_all).max()), 0.1)
    ax2.set_xlim(-1.1 * x_lim, 1.1 * x_lim)
    ax2.set_ylim(-1.1 * y_lim, 1.1 * y_lim)

    ax2.axhline(0, color="#cccccc", linewidth=0.8)
    ax2.axvline(0, color="#cccccc", linewidth=0.8)

    for _, row in df.iterrows():
        ax2.text(
            row["mean_norm"],
            row["diff_norm"],
            " " + row["predictor"],
            fontsize=8,
            ha="left",
            va="center",
        )

    ax2.set_xlabel("Average coefficient (judges & fans)")
    ax2.set_ylabel("Difference (judges - fans), coefficient")
    ax2.set_title("Judges vs fans: agreement vs difference")

    cbar2 = fig2.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2, shrink=0.8
    )
    cbar2.set_label("coef_score - coef_vote (judges minus fans)")

    out_path2 = OUT_DIR / "fig_coef_judges_vs_fans_rotated.png"
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=300)
    plt.close(fig2)
    print(f"Saved: {out_path2}")

    # --- Figure 3: dumbbell plot per predictor ---
    # Horizontal comparison of judges vs fans for each feature
    df_sorted = df.sort_values(
        "diff", key=lambda s: np.abs(s), ascending=False
    ).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))

    fig3, ax3 = plt.subplots(figsize=(7.5, max(4.0, 0.4 * len(df_sorted))))

    # Colors for endpoints
    color_fans = "#ee4d7a"
    color_judges = "#3e7cd8"
    r_f, g_f, b_f = mcolors.to_rgb(color_fans)
    r_j, g_j, b_j = mcolors.to_rgb(color_judges)

    # Gradient line between the two points for each predictor
    n_steps = 40
    for idx, row in df_sorted.iterrows():
        x1 = float(_stretch(np.array([row["coef_score_norm"]]))[0])
        x2 = float(_stretch(np.array([row["coef_vote_norm"]]))[0])
        y = float(y_pos[idx])
        if x1 == x2:
            # If coefficients are (almost) identical, draw a short solid line in mid color
            mid_color = (0.5 * (r_f + r_j), 0.5 * (g_f + g_j), 0.5 * (b_f + b_j))
            ax3.plot(
                [x1 - 1e-3, x2 + 1e-3],
                [y, y],
                color=mid_color,
                linewidth=2.5,
                solid_capstyle="round",
                zorder=1,
            )
            continue
        xs = np.linspace(x1, x2, n_steps)
        for k in range(n_steps - 1):
            # t from 0 at judges (blue) to 1 at fans (pink)
            t = k / (n_steps - 1)
            col = (
                (1 - t) * r_j + t * r_f,
                (1 - t) * g_j + t * g_f,
                (1 - t) * b_j + t * b_f,
            )
            ax3.plot(
                [xs[k], xs[k + 1]],
                [y, y],
                color=col,
                linewidth=2.5,
                solid_capstyle="round",
                zorder=1,
            )

    # Base points (slightly larger) for fans and judges
    s_base = 60
    ax3.scatter(
        _stretch(df_sorted["coef_vote_norm"].to_numpy()),
        y_pos,
        color=color_fans,
        s=s_base,
        label="fans vote",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax3.scatter(
        _stretch(df_sorted["coef_score_norm"].to_numpy()),
        y_pos,
        color=color_judges,
        s=s_base,
        label="judges score",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )

    # Glossy highlight: smaller, semi-transparent white dots on top
    s_highlight = 22
    ax3.scatter(
        _stretch(df_sorted["coef_vote_norm"].to_numpy()),
        y_pos,
        color="white",
        alpha=0.6,
        s=s_highlight,
        linewidth=0,
        zorder=5,
    )
    ax3.scatter(
        _stretch(df_sorted["coef_score_norm"].to_numpy()),
        y_pos,
        color="white",
        alpha=0.6,
        s=s_highlight,
        linewidth=0,
        zorder=5,
    )

    # Grey reference line at coefficient 0 (after stretching)
    zero_view_3 = _stretch(np.array([0.0]))[0]
    ax3.axvline(zero_view_3, color="#cccccc", linewidth=0.8)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(df_sorted["predictor"])

    # Use the same stretched-domain limits as the main scatter
    x_lim_view = _stretch(np.array([lo_raw, hi_raw]))
    ax3.set_xlim(x_lim_view.min(), x_lim_view.max())

    # Reuse the same tick positions and labels so the numeric
    # values on the axis remain the original coefficients
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels([f"{v:.2g}" for v in tick_vals])
    ax3.invert_yaxis()

    ax3.set_xlabel("Coefficient")
    ax3.set_title("Judges Score vs Fans Vote: Coefficient Comparison By Feature")
    ax3.legend(loc="lower right", fontsize=8, frameon=False)

    out_path3 = OUT_DIR / "fig_coef_judges_vs_fans_dumbbell.png"
    fig3.tight_layout()
    fig3.savefig(out_path3, dpi=300)
    plt.close(fig3)
    print(f"Saved: {out_path3}")


if __name__ == "__main__":
    main()
