"""Generate a gallery of publication-friendly matplotlib figures from celebrity_features_summary.csv.

Outputs are written to outputs/ under filenames prefixed with fig_. The goal is variety; pick what you need.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D plots


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "celebrity_features_summary.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Consistent, clean styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
COLOR_CYCLE = plt.get_cmap("tab10").colors
ACCENT = "#2b6cb0"  # deep blue
# Custom palette for the pie chart / industry colors
PIE_PALETTE = [
    "#3e7cd8",  # blue
    "#9C86DA",  # purple
    "#ee4d7a",  # pink
    "#EF482D",  # orange-red
]

INDUSTRY_COLOR_MAP = {
    "Athletic": "#3e7cd8",
    "Performance": "#9C86DA",
    "Media": "#ee4d7a",
    "Other": "#EF482D",
}


def _save(fig, name: str):
    path = OUT_DIR / f"fig_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Basic safety: drop obvious missing columns silently
    required_cols = {
        "season",
        "celebrity_name",
        "industry_group",
        "placement_encoded",
        "avg_score_rate",
        "avg_voting_rate",
        "base_popularity",
        "celebrity_homestate",
        "is_us",
        "age",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Warning: missing columns {missing}; some plots may be skipped.")

    # Pie: industry composition (purple/pink palette)
    if {"industry_group"}.issubset(df.columns):
        industry_counts = (
            df["industry_group"].value_counts().sort_values(ascending=False)
        )
        if len(industry_counts) >= 1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                industry_counts,
                labels=industry_counts.index,
                colors=PIE_PALETTE,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
            )
            ax.set_title("Industry composition")
            _save(fig, "pie_industry")

    # Pie: age bucket composition (young / mid / senior / unknown)
    if "age_bucket" in df.columns or "age" in df.columns:
        if "age_bucket" not in df.columns:
            # Fallback: derive age buckets from raw age if needed
            age_series = df["age"].copy()

            def _age_bucket(x):
                if pd.isna(x):
                    return "unknown"
                if x < 30:
                    return "young"
                if x <= 55:
                    return "mid"
                return "senior"

            df["age_bucket"] = age_series.apply(_age_bucket)

        age_order = ["young", "mid", "senior", "unknown"]
        age_counts = (
            df["age_bucket"].value_counts().reindex(age_order).dropna().astype(int)
        )
        if len(age_counts) > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = PIE_PALETTE * ((len(age_counts) // len(PIE_PALETTE)) + 1)
            ax.pie(
                age_counts,
                labels=age_counts.index,
                colors=colors[: len(age_counts)],
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
            )
            ax.set_title("Age bucket composition")
            _save(fig, "pie_age_bucket")

    # 5) Scatter: popularity baseline vs placement
    if {"base_popularity", "placement_encoded", "industry_group"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        clean = df.dropna(subset=["base_popularity", "placement_encoded"])
        for ind, sub in clean.groupby("industry_group"):
            if sub.empty:
                continue
            color = INDUSTRY_COLOR_MAP.get(str(ind), "#999999")
            ax.scatter(
                sub["base_popularity"],
                sub["placement_encoded"],
                s=30,
                alpha=0.7,
                label=str(ind),
                color=color,
            )
        ax.set_xlabel("base popularity")
        ax.set_ylabel("placement encoded")
        ax.set_title("Popularity vs Final Placement")
        if clean.shape[0]:
            ax.legend(frameon=True)
        _save(fig, "scatter_pop_vs_place")

    # 6) Heatmap: season x industry mean placement
    if {"season", "industry_group", "placement_encoded"}.issubset(df.columns):
        pivot = df.pivot_table(
            index="season",
            columns="industry_group",
            values="placement_encoded",
            aggfunc="mean",
        ).sort_index()
        # Drop all-NaN rows/cols to avoid blank heatmaps
        pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            im = ax.imshow(pivot, aspect="auto", cmap="YlGnBu")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Industry")
            ax.set_ylabel("Season")
            ax.set_title("Mean placement by season and industry")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean placement (encoded)")
            _save(fig, "heatmap_season_industry")

    # 9) Stacked bar: US vs non-US by season
    if {"season", "is_us"}.issubset(df.columns):
        by_season = df.groupby(["season", "is_us"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        by_season.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color={0: "#ee4d7a", 1: "#3e7cd8"},
            width=0.85,
        )
        ax.set_xlabel("Season")
        ax.set_ylabel("Count")
        ax.set_title("US vs non-US Celebrities By Season")
        # Keep season tick labels upright
        ax.tick_params(axis="x", labelrotation=0)
        ax.legend(labels=["Non-US", "US"], title="Origin")
        _save(fig, "bar_stacked_us_nonus")

    # Coefficient-based visualizations
    coeff_path = OUT_DIR / "mixedlm_results.json"
    if coeff_path.exists():
        try:
            with open(coeff_path, "r", encoding="utf-8") as f:
                models = json.load(f)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to read coefficients: {exc}")
            models = {}

        key_order = [
            "age_young",
            "age_mid",
            "age_senior",
            "is_us",
            "pro_strength",
            "base_popularity",
            "indgrp_Athletic",
            "indgrp_Performance",
            "indgrp_Media",
            "indgrp_Other",
        ]

        targets = ["avg_score_rate", "avg_voting_rate", "placement_encoded"]
        data = []
        for tgt in targets:
            params = (
                models.get(tgt, {}).get("params", {})
                if isinstance(models.get(tgt), dict)
                else {}
            )
            for k in params:
                data.append({"predictor": k, "target": tgt, "coef": params[k]})

        if data:
            coef_df = pd.DataFrame(data)
            predictors = [k for k in key_order if k in coef_df["predictor"].unique()]
            if not predictors:
                return

            # 1) Line plot for placement_encoded: main vs even-odd models
            main_params = (
                models.get("placement_encoded", {}).get("params", {})
                if isinstance(models.get("placement_encoded"), dict)
                else {}
            )
            even_params = (
                models.get("placement_encoded_evenodd", {}).get("params", {})
                if isinstance(models.get("placement_encoded_evenodd"), dict)
                else {}
            )

            if main_params:
                y_main = [main_params.get(p, np.nan) for p in predictors]
                y_even = [even_params.get(p, np.nan) for p in predictors]
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(
                    predictors,
                    y_main,
                    marker="o",
                    label="main",
                    color="#2b6cb0",
                    linewidth=2.0,
                )
                if any(np.isfinite(y_even)):
                    ax.plot(
                        predictors,
                        y_even,
                        marker="o",
                        label="even-odd",
                        color="#9cb3e5",
                        linewidth=1.6,
                    )

                all_vals = np.array(y_main + y_even, dtype=float)
                all_vals = all_vals[np.isfinite(all_vals)]
                if all_vals.size:
                    vmin, vmax = float(all_vals.min()), float(all_vals.max())
                    margin = 0.1 * (vmax - vmin if vmax > vmin else max(abs(vmax), 1.0))
                    ax.set_ylim(vmin - margin, vmax + margin)

                ax.set_xlabel("Predictor")
                ax.set_ylabel("Coefficient")
                ax.set_title(
                    "Placement-encoded Coefficients In Main And Even-Odd Models"
                )
                ax.legend(title="Model")
                ax.tick_params(axis="x", rotation=45)
                _save(fig, "line_coeff_placement")

            # 2) 3D surface for avg_score_rate & avg_voting_rate coefficients
            targets_2d = ["avg_score_rate", "avg_voting_rate"]
            # Build grid: X dimension = target (2 columns), Y dimension = predictors
            x_idx = np.arange(len(targets_2d))
            y_idx = np.arange(len(predictors))
            X, Y = np.meshgrid(x_idx, y_idx)
            Z = np.full_like(X, np.nan, dtype=float)

            for j, tgt in enumerate(targets_2d):
                sub = coef_df[coef_df["target"] == tgt].set_index("predictor")
                for i, p in enumerate(predictors):
                    if p in sub.index:
                        Z[i, j] = sub.loc[p, "coef"]

            if np.isfinite(Z).any():
                Z_plot = np.ma.array(Z, mask=~np.isfinite(Z))
                fig = plt.figure(figsize=(9, 6))
                ax3d = fig.add_subplot(111, projection="3d")
                surf = ax3d.plot_surface(
                    X,
                    Y,
                    Z_plot,
                    cmap="coolwarm",
                    edgecolor="none",
                    antialiased=True,
                    alpha=0.9,
                )
                fig.colorbar(surf, ax=ax3d, shrink=0.6, aspect=12, label="Coefficient")

                # Highlight top-3 absolute coefficients across these two targets
                abs_z = np.abs(Z)
                mask = np.isfinite(abs_z)
                if mask.any():
                    flat_idx = np.argsort(abs_z[mask])[-3:]
                    yi, xi = np.where(mask)
                    top_y = yi[np.argsort(abs_z[mask])[-3:]]
                    top_x = xi[np.argsort(abs_z[mask])[-3:]]
                    for tx, ty in zip(top_x, top_y):
                        ax3d.scatter(
                            tx,
                            ty,
                            Z[ty, tx],
                            color="k",
                            s=35,
                            depthshade=False,
                        )
                        feat_name = predictors[ty]
                        ax3d.text(
                            tx,
                            ty,
                            Z[ty, tx],
                            f" {feat_name}",
                            fontsize=8,
                            color="k",
                        )

                ax3d.set_xticks(x_idx)
                ax3d.set_xticklabels(
                    [
                        "avg_score_rate",
                        "avg_voting_rate",
                    ]
                )
                ax3d.set_yticks(y_idx)
                ax3d.set_yticklabels(predictors, fontsize=8)
                ax3d.set_xlabel("Target variable")
                ax3d.set_ylabel("Predictor")
                ax3d.set_zlabel("Coefficient")
                ax3d.set_title(
                    "Coefficient surface for avg_score_rate and avg_voting_rate"
                )
                ax3d.view_init(elev=35, azim=-45)
                _save(fig, "surface_coef_score_vote")


if __name__ == "__main__":
    main()
