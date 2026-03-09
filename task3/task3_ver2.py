"""Feature extraction for Problem 3 (celebrity-level features).

This script:
- Loads Cleaned_data_with_votes.csv
- Merges task1_ultimate_popularity.csv to use base popularity as a prior
- Cleans partner names
- Builds per-celebrity season-level features (technical vs popularity); industry采用独热编码
- Fits mixed-effects models for avg_score_rate, avg_voting_rate, placement_encoded (排名越大越好)
- Saves celebrity_features_summary.csv to outputs/
- Generates static PNG plots (if matplotlib/seaborn available) and a US state choropleth (HTML) counting celebrities by home state

Note: Visualizations are lightweight; static PNGs are generated only when matplotlib/seaborn are installed.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cleaned_data_with_votes.csv"
POP_PATH = BASE_DIR / "task1_ultimate_popularity.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Merge baseline popularity from task1 output
if POP_PATH.exists():
    df_pop = pd.read_csv(POP_PATH)
    df = df.merge(df_pop, on=["season", "celebrity_name"], how="left")
else:  # pragma: no cover - file missing is unexpected
    df_pop = pd.DataFrame()

# Normalize column name with slash for easier handling
df = df.rename(columns={"celebrity_homecountry/region": "celebrity_homecountry_region"})

# Ensure voting_rate exists (fallback to final_est_share if present)
if "voting_rate" not in df.columns and "final_est_share" in df.columns:
    df["voting_rate"] = df["final_est_share"]

# Popularity baseline (prefer task1_ultimate_popularity output)
if "base_popularity_share" in df.columns:
    df["base_popularity"] = df["base_popularity_share"]
elif "base_popularity" in df.columns:
    df["base_popularity"] = df["base_popularity"]
else:
    df["base_popularity"] = np.nan

df["base_popularity"] = pd.to_numeric(df["base_popularity"], errors="coerce")
df["placement"] = pd.to_numeric(df["placement"], errors="coerce")

# Clean partner names: drop trailing parenthetical notes like " (Week X with Y)"
df["ballroom_partner"] = df["ballroom_partner"].str.replace(
    r"\s*\(.*?\)$", "", regex=True
)

# Industry grouping -> four buckets to reduce dimensionality
IND_GROUP_LEVELS = ["Athletic", "Performance", "Media", "Other"]

_ATHLETIC = {
    "athlete",
    "racing driver",
    "fitness instructor",
    "astronaut",
    "military",
}
_PERFORMANCE = {
    "actor/actress",
    "singer/rapper",
    "musician",
    "comedian",
    "model",
    "magician",
    "fashion designer",
    "beauty pagent",
    "beauty pageant",
}
_MEDIA = {
    "tv personality",
    "news anchor",
    "sports broadcaster",
    "radio personality",
    "journalist",
    "social media personality",
    "motivational speaker",
}


def _industry_group(raw: str) -> str:
    if pd.isna(raw):
        return "Other"
    key = str(raw).strip().lower()
    # Normalize a common variant
    key = key.replace("social media personality", "social media personality")
    if key in _ATHLETIC:
        return "Athletic"
    if key in _PERFORMANCE:
        return "Performance"
    if key in _MEDIA:
        return "Media"
    return "Other"


df["industry_group"] = df["celebrity_industry"].apply(_industry_group)
df["industry_group"] = pd.Categorical(df["industry_group"], categories=IND_GROUP_LEVELS)


# -----------------------------
# Basic feature engineering at the row (week) level
# -----------------------------
df["age"] = df["celebrity_age_during_season"]


# Age bucket -> one-hot later: young / mid / senior
AGE_LEVELS = ["young", "mid", "senior"]


def _age_bucket(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x < 30:
        return "young"
    if x <= 55:
        return "mid"
    return "senior"


df["age_bucket"] = df["age"].apply(_age_bucket)

# Country flags
df["is_us"] = (
    df["celebrity_homecountry_region"]
    .fillna("")
    .str.contains("United States", case=False)
    .astype(int)
)
df["is_non_us"] = 1 - df["is_us"]

# Season max week for handling 999 sentinel
season_max_week = df.groupby("season")["week"].max()
df = df.merge(season_max_week.rename("season_max_week"), on="season", how="left")
df["last_week_active"] = np.where(
    df["last_active_week"] == 999, df["season_max_week"], df["last_active_week"]
)

# Pro dancer strength: mean score_rate across all seasons
partner_strength = (
    df.groupby("ballroom_partner")["score_rate"]
    .mean()
    .reset_index()
    .rename(columns={"score_rate": "pro_strength"})
)
df = df.merge(partner_strength, on="ballroom_partner", how="left")


# -----------------------------
# Per-celebrity (per season) aggregation
# -----------------------------
def _last_scored_week(sub_df: pd.DataFrame) -> int:
    scored_weeks = sub_df.loc[sub_df["total_judge_score"] > 0, "week"]
    return scored_weeks.max() if len(scored_weeks) else sub_df["week"].max()


grouped = df.groupby(["season", "celebrity_name"], as_index=False)

celeb_features = grouped.agg(
    celebrity_industry=("celebrity_industry", "first"),
    industry_group=("industry_group", "first"),
    age=("age", "first"),
    age_bucket=("age_bucket", "first"),
    celebrity_homecountry_region=("celebrity_homecountry_region", "first"),
    celebrity_homestate=("celebrity_homestate", "first"),
    is_us=("is_us", "first"),
    ballroom_partner=("ballroom_partner", "first"),
    pro_strength=("pro_strength", "first"),
    avg_score_rate=("score_rate", "mean"),
    avg_voting_rate=("voting_rate", "mean"),
    base_popularity=("base_popularity", "mean"),
    placement=("placement", "min"),
)

# Placement encoding: linear steps from 1 down to ~1/n (e.g., 5 人季节 -> 1,0.8,0.6,0.4,0.2)
celeb_features["season_size"] = celeb_features.groupby("season")[
    "celebrity_name"
].transform("count")
celeb_features["placement_encoded"] = np.where(
    celeb_features["season_size"] > 0,
    1.0 - (celeb_features["placement"] - 1) / celeb_features["season_size"],
    np.nan,
)

# Recompute pro_strength: blend avg_score_rate 与 placement_encoded (min-max 归一化后取均值)
partner_stats = celeb_features.groupby("ballroom_partner").agg(
    score_mean=("avg_score_rate", "mean"),
    place_mean=("placement_encoded", "mean"),
)


def _minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


partner_stats["score_norm"] = _minmax(partner_stats["score_mean"])
partner_stats["place_norm"] = _minmax(partner_stats["place_mean"])
partner_stats["pro_strength_new"] = (
    0.5 * partner_stats["score_norm"] + 0.5 * partner_stats["place_norm"]
)

celeb_features = celeb_features.drop(columns=["pro_strength"], errors="ignore")
celeb_features = celeb_features.merge(
    partner_stats[["pro_strength_new"]],
    left_on="ballroom_partner",
    right_index=True,
    how="left",
)
celeb_features = celeb_features.rename(columns={"pro_strength_new": "pro_strength"})

# Trim helper columns if present
celeb_features = celeb_features.drop(
    columns=[col for col in ["index", "Unnamed: 20"] if col in celeb_features.columns]
)

# Ensure industry categories are consistent across splits
celeb_features["industry_group"] = pd.Categorical(
    celeb_features["industry_group"], categories=IND_GROUP_LEVELS
)

# Age one-hot encoding (young/mid/senior); unknown -> all zeros
age_dummies = pd.get_dummies(
    celeb_features["age_bucket"], prefix="age", drop_first=False
).reindex(columns=["age_young", "age_mid", "age_senior"], fill_value=0)
age_dummies = age_dummies.fillna(0).astype(int)

# Industry one-hot encoding (four groups)
ind_dummies = pd.get_dummies(
    celeb_features["industry_group"], prefix="indgrp", drop_first=False
).reindex(columns=[f"indgrp_{g}" for g in IND_GROUP_LEVELS], fill_value=0)
ind_dummies = ind_dummies.fillna(0).astype(int)

celeb_features = pd.concat([celeb_features, age_dummies, ind_dummies], axis=1)

# 记录行业表现：按 placement_encoded 均值排序，便于后续提取“强势行业”
top_industries = (
    celeb_features.groupby("industry_group")["placement_encoded"]
    .mean()
    .sort_values(ascending=False)
)


# -----------------------------
# State-level heatmap (USA) based on celebrity home states
# --------------------------------------------------------
STATE_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}

# Approximate state centroids for on-map text placement
STATE_CENTROIDS = {
    "AL": (32.806671, -86.791130),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564),
    "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),
    "DE": (39.318523, -75.507141),
    "DC": (38.907192, -77.036873),
    "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337),
    "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),
    "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526),
    "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067),
    "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927),
    "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106),
    "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),
    "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368),
    "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082),
    "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482),
    "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780),
    "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),
    "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686),
    "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494),
    "WV": (38.491226, -80.954456),
    "WI": (44.268543, -89.616508),
    "WY": (42.755966, -107.302490),
}

state_base = (
    celeb_features.loc[celeb_features["is_us"] == 1]
    .assign(
        state_upper=lambda d: d["celebrity_homestate"]
        .fillna("")
        .str.upper()
        .str.strip()
    )
    .assign(state_abbr=lambda d: d["state_upper"].map(STATE_ABBR))
    .dropna(subset=["state_abbr"])
)

state_stats = (
    state_base.groupby("state_abbr")
    .agg(count=("state_abbr", "size"), mean_final_rank=("placement", "mean"))
    .reset_index()
)

state_counts = state_stats[["state_abbr", "count"]]

if not state_counts.empty:
    fig = px.choropleth(
        state_counts,
        locations="state_abbr",
        locationmode="USA-states",
        color="count",
        scope="usa",
        color_continuous_scale="YlOrRd",
        title="Number of celebrities by home state",
    )

    # Annotate states with >=10 celebrities using mean final placement
    annotate_df = state_stats[state_stats["count"] >= 10].copy()
    annotate_df["mean_final_rank"] = annotate_df["mean_final_rank"].astype(float)
    annotate_df["lon"] = annotate_df["state_abbr"].map(
        lambda abbr: STATE_CENTROIDS.get(abbr, (np.nan, np.nan))[1]
    )
    annotate_df["lat"] = annotate_df["state_abbr"].map(
        lambda abbr: STATE_CENTROIDS.get(abbr, (np.nan, np.nan))[0]
    )
    annotate_df = annotate_df.dropna(subset=["lat", "lon", "mean_final_rank"])

    if not annotate_df.empty:
        fig.add_scattergeo(
            lon=annotate_df["lon"],
            lat=annotate_df["lat"],
            mode="text",
            text=[f"{v:.1f}" for v in annotate_df["mean_final_rank"]],
            textfont=dict(color="black", size=12),
            showlegend=False,
            hovertemplate="State: %{location}<br>Mean final rank: %{text}<extra></extra>",
            locationmode="USA-states",
        )
        fig.add_annotation(
            x=0.5,
            y=-0.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="States with numbers have >=10 celebrities; number is mean final placement (lower is better).",
            font=dict(size=12),
        )

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=40))

    # Write both interactive HTML and publication-friendly PNG (requires kaleido)
    fig.write_html(OUT_DIR / "us_state_counts.html")
    try:
        fig.write_image(
            OUT_DIR / "us_state_counts.png", width=1200, height=800, scale=2
        )
    except Exception as exc:
        print(f"Failed to write PNG (install kaleido?): {exc}")


# -----------------------------
# Mixed-effects models for avg_score_rate and avg_voting_rate
# -----------------------------
import statsmodels.formula.api as smf


AGE_DUMMY_COLS = ["age_young", "age_mid", "age_senior"]
IND_DUMMY_COLS = [f"indgrp_{g}" for g in IND_GROUP_LEVELS]
PREDICTORS = (
    AGE_DUMMY_COLS
    + [
        "is_us",
        "pro_strength",
        "base_popularity",
    ]
    + IND_DUMMY_COLS
)


def _train_test_split_by_season(df_in: pd.DataFrame, test_frac: float = 0.2):
    seasons = sorted(df_in["season"].unique())
    n_test = max(1, int(len(seasons) * test_frac))
    test_seasons = set(seasons[-n_test:])
    train_seasons = set(seasons[:-n_test])
    train_df = df_in[df_in["season"].isin(train_seasons)].copy()
    test_df = df_in[df_in["season"].isin(test_seasons)].copy()
    return train_df, test_df, test_seasons


def fit_mixedlm(df_in: pd.DataFrame, response: str):
    usable = df_in.dropna(subset=[response, *PREDICTORS])
    train_df, test_df, test_seasons = _train_test_split_by_season(usable)
    rhs = " + ".join(PREDICTORS)
    formula = f"{response} ~ -1 + {rhs}"

    def _rmse(y_true, y_pred):
        return (
            float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan
        )

    mixed_error = None
    # 对 avg_score_rate 强制使用 OLS，避免混合效应数值不稳导致极端预测
    if response == "avg_score_rate":
        ols_model = smf.ols(formula, data=train_df)
        result = ols_model.fit()
        model_type = "ols_forced"
    else:
        try:
            model = smf.mixedlm(formula, data=train_df, groups=train_df["season"])
            result = model.fit(reml=True, method="lbfgs")
            model_type = "mixedlm"
        except Exception as exc:
            mixed_error = str(exc)
            ols_model = smf.ols(formula, data=train_df)
            result = ols_model.fit()
            model_type = "ols_fallback"

    train_rmse = _rmse(train_df[response], result.predict(train_df))
    test_rmse = (
        _rmse(test_df[response], result.predict(test_df)) if len(test_df) else np.nan
    )
    return result, train_rmse, test_rmse, test_seasons, model_type, mixed_error


def _regression_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    response: str,
    model,
    clip_bounds: tuple | None = None,
) -> dict:
    """计算回归评估指标：RMSE / MAE / R^2 以及仅预测训练均值的基线 RMSE。

    说明：
    - baseline 使用训练集均值作为常数预测，对训练集和测试集同时评估；
    - R^2 按 1 - MSE/Var(y) 计算，当 Var(y)<=0 时返回 NaN。
    """

    def _rmse_arr(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0:
            return np.nan
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _mae_arr(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0:
            return np.nan
        return float(np.mean(np.abs(y_true - y_pred)))

    # 真实值与预测值
    y_tr = train_df[response]
    yhat_tr = model.predict(train_df)
    y_te = test_df[response] if len(test_df) else pd.Series(dtype=float)
    yhat_te = model.predict(test_df) if len(test_df) else pd.Series(dtype=float)

    # 若指定 clip_bounds，则将预测值裁剪到合理区间（如 0-1 的比例变量）
    if clip_bounds is not None:
        lo, hi = clip_bounds
        yhat_tr = np.clip(yhat_tr, lo, hi)
        yhat_te = np.clip(yhat_te, lo, hi)

    # 基线：训练集均值
    if len(y_tr):
        baseline_value = float(y_tr.mean())
    else:
        baseline_value = np.nan

    baseline_tr_rmse = _rmse_arr(y_tr, baseline_value) if len(y_tr) else np.nan
    baseline_te_rmse = _rmse_arr(y_te, baseline_value) if len(y_te) else np.nan

    # 模型 RMSE / MAE
    train_rmse = _rmse_arr(y_tr, yhat_tr)
    test_rmse = _rmse_arr(y_te, yhat_te) if len(y_te) else np.nan
    train_mae = _mae_arr(y_tr, yhat_tr)
    test_mae = _mae_arr(y_te, yhat_te) if len(y_te) else np.nan

    # R^2（当方差不为 0 时才计算）
    def _r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size <= 1:
            return np.nan
        var = float(np.var(y_true))
        if var <= 0:
            return np.nan
        mse = float(np.mean((y_true - y_pred) ** 2))
        return float(1.0 - mse / var)

    r2_train = _r2(y_tr, yhat_tr)
    r2_test = _r2(y_te, yhat_te) if len(y_te) else np.nan

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "baseline_train_rmse": baseline_tr_rmse,
        "baseline_test_rmse": baseline_te_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }


models_output = {}
for target in [
    "avg_score_rate",
    "avg_voting_rate",
    "placement_encoded",
]:
    try:
        res, tr_rmse, te_rmse, te_seasons, model_type, mixed_error = fit_mixedlm(
            celeb_features, target
        )

        # 基于与建模完全一致的划分重新获取 train/test 数据，用于统一评估指标
        usable_main = celeb_features.dropna(subset=[target, *PREDICTORS])
        train_main, test_main, _ = _train_test_split_by_season(usable_main)
        bounds = (
            (0.0, 1.0)
            if target in ["avg_score_rate", "avg_voting_rate", "placement_encoded"]
            else None
        )
        metric_main = _regression_metrics(
            train_main, test_main, target, res, clip_bounds=bounds
        )

        models_output[target] = {
            "params": res.params.to_dict(),
            "train_rmse": metric_main["train_rmse"],
            "test_rmse": metric_main["test_rmse"],
            "test_seasons": sorted(list(te_seasons)),
            "split": "chronological_last20pct",
            "model_type": model_type,
            # 额外评估指标
            "baseline_train_rmse": metric_main["baseline_train_rmse"],
            "baseline_test_rmse": metric_main["baseline_test_rmse"],
            "train_mae": metric_main["train_mae"],
            "test_mae": metric_main["test_mae"],
            "r2_train": metric_main["r2_train"],
            "r2_test": metric_main["r2_test"],
        }
        if mixed_error:
            models_output[target]["mixedlm_error"] = mixed_error
    except Exception as exc:  # pragma: no cover - modeling is best-effort
        models_output[target] = {"error": str(exc)}


# Additional split: even vs odd seasons (robustness)
def _train_test_even_odd(df_in: pd.DataFrame):
    seasons = df_in["season"].unique()
    train_df = df_in[df_in["season"].isin([s for s in seasons if s % 2 == 0])]
    test_df = df_in[df_in["season"].isin([s for s in seasons if s % 2 == 1])]
    return train_df, test_df, sorted(list(test_df["season"].unique()))


for target in ["avg_score_rate", "avg_voting_rate", "placement_encoded"]:
    try:
        usable = celeb_features.dropna(subset=[target, *PREDICTORS])
        train_df, test_df, te_seasons = _train_test_even_odd(usable)
        rhs = " + ".join(PREDICTORS)
        formula = f"{target} ~ -1 + {rhs}"

        def _rmse(y_true, y_pred):
            return (
                float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                if len(y_true)
                else np.nan
            )

        mixed_error = None
        if target == "avg_score_rate":
            # 对 avg_score_rate 强制使用 OLS，避免混合效应不稳定
            ols_model = smf.ols(formula, data=train_df)
            result = ols_model.fit()
            model_type = "ols_forced"
        else:
            try:
                model = smf.mixedlm(formula, data=train_df, groups=train_df["season"])
                result = model.fit(reml=True, method="lbfgs")
                model_type = "mixedlm"
            except Exception as exc_inner:
                mixed_error = str(exc_inner)
                ols_model = smf.ols(formula, data=train_df)
                result = ols_model.fit()
                model_type = "ols_fallback"

        train_rmse = _rmse(train_df[target], result.predict(train_df))
        test_rmse = (
            _rmse(test_df[target], result.predict(test_df)) if len(test_df) else np.nan
        )
        bounds = (
            (0.0, 1.0)
            if target in ["avg_score_rate", "avg_voting_rate", "placement_encoded"]
            else None
        )
        metric_eo = _regression_metrics(
            train_df, test_df, target, result, clip_bounds=bounds
        )
        models_output[f"{target}_evenodd"] = {
            "params": result.params.to_dict(),
            "train_rmse": metric_eo["train_rmse"],
            "test_rmse": metric_eo["test_rmse"],
            "test_seasons": te_seasons,
            "split": "even_train_vs_odd_test",
            "model_type": model_type,
            "baseline_train_rmse": metric_eo["baseline_train_rmse"],
            "baseline_test_rmse": metric_eo["baseline_test_rmse"],
            "train_mae": metric_eo["train_mae"],
            "test_mae": metric_eo["test_mae"],
            "r2_train": metric_eo["r2_train"],
            "r2_test": metric_eo["r2_test"],
        }
        if mixed_error:
            models_output[f"{target}_evenodd"]["mixedlm_error"] = mixed_error
    except Exception as exc:  # pragma: no cover
        models_output[f"{target}_evenodd"] = {"error": str(exc)}


# -----------------------------
# Visualizations - 仅保留美国地图 HTML
# --------------------------------------------------------


# -----------------------------
# Save tables and model outputs
# -----------------------------
celeb_features.to_csv(OUT_DIR / "celebrity_features_summary.csv", index=False)
pd.Series(models_output).to_json(OUT_DIR / "mixedlm_results.json")


def _rmse_comment(train_rmse: float, test_rmse: float) -> str:
    """根据训练 / 测试 RMSE 的大小关系，给出一句中文评价。"""

    if np.isnan(test_rmse):
        return "测试集为空，暂时无法判断泛化能力。"
    if train_rmse <= 0 or np.isnan(train_rmse):
        return "训练误差异常（为 0 或缺失），请手动检查结果。"

    ratio = test_rmse / train_rmse
    if ratio <= 1.1:
        return "测试误差与训练误差非常接近，模型泛化能力较好。"
    if ratio <= 1.3:
        return "测试误差略高于训练误差，存在一定程度的过拟合，但整体尚可。"
    return "测试误差明显高于训练误差，存在较明显过拟合，需要谨慎解读。"


def print_model_summary_zh(models_dict: dict) -> None:
    """将混合效应模型的结果用中文较整洁地打印到终端。"""

    target_name = {
        "avg_score_rate": "平均评委得分率 (avg_score_rate)",
        "avg_voting_rate": "平均粉丝投票占比 (avg_voting_rate)",
        "placement_encoded": "最终排名编码 (placement_encoded，数值越大越好)",
    }
    split_desc = {
        "chronological_last20pct": "按季时间顺序划分：最后 20% 的季作为测试集",
        "even_train_vs_odd_test": "偶数季训练、奇数季测试（稳健性检验）",
    }

    print("\n================= 混合效应模型结果概览 =================")
    for base in ["avg_score_rate", "avg_voting_rate", "placement_encoded"]:
        print(f"\n>>> 目标变量：{target_name.get(base, base)}")

        # 主划分（时间顺序）
        main_res = models_dict.get(base)
        if main_res is None:
            print("  - 未成功拟合主模型。")
        elif "error" in main_res:
            print(f"  - 主模型拟合失败：{main_res['error']}")
        else:
            tr = float(main_res.get("train_rmse", np.nan))
            te = float(main_res.get("test_rmse", np.nan))
            sp = main_res.get("split", "")
            model_type = main_res.get("model_type", "mixedlm")
            base_tr = float(main_res.get("baseline_train_rmse", np.nan))
            base_te = float(main_res.get("baseline_test_rmse", np.nan))
            mae_tr = float(main_res.get("train_mae", np.nan))
            mae_te = float(main_res.get("test_mae", np.nan))
            r2_tr = float(main_res.get("r2_train", np.nan))
            r2_te = float(main_res.get("r2_test", np.nan))
            print("  [主模型]")
            print(f"  - 划分方式：{split_desc.get(sp, sp)}")
            print(f"  - 训练集 RMSE：{tr:.3f}，测试集 RMSE：{te:.3f}")
            if not np.isnan(base_tr) or not np.isnan(base_te):
                print(
                    f"  - 基线 RMSE（仅预测训练均值）：训练 {base_tr:.3f}，测试 {base_te:.3f}"
                )
            if not np.isnan(mae_tr) or not np.isnan(mae_te):
                print(f"  - MAE：训练 {mae_tr:.3f}，测试 {mae_te:.3f}")
            if not np.isnan(r2_tr) or not np.isnan(r2_te):
                print(f"  - R²：训练 {r2_tr:.3f}，测试 {r2_te:.3f}")
            print(f"  - 误差评价：{_rmse_comment(tr, te)}")
            if model_type != "mixedlm":
                if model_type == "ols_forced":
                    print("  - 拟合方式：ols_forced（该目标直接使用 OLS）")
                else:
                    print(f"  - 拟合方式：{model_type}（混合效应失败后回退）")
                    if "mixedlm_error" in main_res:
                        print(f"    混合效应失败原因：{main_res['mixedlm_error']}")

            params = main_res.get("params", {})
            if params:
                # 全量系数，按绝对值排序，避免遗漏
                sorted_params = sorted(
                    params.items(), key=lambda kv: abs(kv[1]), reverse=True
                )
                print("  - 主要系数（按绝对值从大到小列出全部）：")
                for name, val in sorted_params:
                    print(f"      {name:<40s} = {val: .3f}")

        # 奇偶季划分结果（稳健性检查）
        eo_res = models_dict.get(f"{base}_evenodd")
        if eo_res is None:
            continue
        if "error" in eo_res:
            print(f"  [奇偶季稳健性模型] 拟合失败：{eo_res['error']}")
        else:
            tr = float(eo_res.get("train_rmse", np.nan))
            te = float(eo_res.get("test_rmse", np.nan))
            sp = eo_res.get("split", "even_train_vs_odd_test")
            model_type = eo_res.get("model_type", "mixedlm")
            base_tr = float(eo_res.get("baseline_train_rmse", np.nan))
            base_te = float(eo_res.get("baseline_test_rmse", np.nan))
            mae_tr = float(eo_res.get("train_mae", np.nan))
            mae_te = float(eo_res.get("test_mae", np.nan))
            r2_tr = float(eo_res.get("r2_train", np.nan))
            r2_te = float(eo_res.get("r2_test", np.nan))
            print("  [奇偶季稳健性模型]")
            print(f"  - 划分方式：{split_desc.get(sp, sp)}")
            print(f"  - 训练集 RMSE：{tr:.3f}，测试集 RMSE：{te:.3f}")
            if not np.isnan(base_tr) or not np.isnan(base_te):
                print(
                    f"  - 基线 RMSE（仅预测训练均值）：训练 {base_tr:.3f}，测试 {base_te:.3f}"
                )
            if not np.isnan(mae_tr) or not np.isnan(mae_te):
                print(f"  - MAE：训练 {mae_tr:.3f}，测试 {mae_te:.3f}")
            if not np.isnan(r2_tr) or not np.isnan(r2_te):
                print(f"  - R²：训练 {r2_tr:.3f}，测试 {r2_te:.3f}")
            print(f"  - 误差评价：{_rmse_comment(tr, te)}")
            if model_type != "mixedlm":
                if model_type == "ols_forced":
                    print("  - 拟合方式：ols_forced（该目标直接使用 OLS）")
                else:
                    print(f"  - 拟合方式：{model_type}（混合效应失败后回退）")
                    if "mixedlm_error" in eo_res:
                        print(f"    混合效应失败原因：{eo_res['mixedlm_error']}")
            params = eo_res.get("params", {})
            if params:
                sorted_params = sorted(
                    params.items(), key=lambda kv: abs(kv[1]), reverse=True
                )
                print("  - 主要系数（按绝对值从大到小列出全部）：")
                for name, val in sorted_params:
                    print(f"      {name:<40s} = {val: .3f}")

    print(
        "\n说明：完整的参数字典已保存到 outputs/mixedlm_results.json，可用于论文表格排版。"
    )


print("Artifacts written to:", OUT_DIR)
print("Top industries by placement_encoded (descending):")
for name, val in top_industries.head(5).items():
    print(f"  {name}: {val:.3f}")
print_model_summary_zh(models_output)
