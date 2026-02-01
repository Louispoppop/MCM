import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==================== Configuration & Style ====================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# Color Palette
COLOR_SAFE = "#3e7cd8"      # Blue
COLOR_ELIM = "#ee4d7a"      # Pink/Red
COLOR_JERRY = "#ee4d7a"     # Pink for Jerry (Controversial)
COLOR_DREW = "#3e7cd8"      # Blue for Drew (winner)
COLOR_PURPLE = "#9C86DA"    # Purple for others

# Output Directory
BASE_DIR = r'd:\美赛'
DATA_FILE = os.path.join(BASE_DIR, 'task1_ultimate_estimates.csv')


def load_and_prep_data():
    if not os.path.exists(DATA_FILE):
        print(
            f"Error: {DATA_FILE} not found. Please run task1_ver8_ultimate.py first.")
        exit()

    df = pd.read_csv(DATA_FILE)

    # Calculate CV
    # Avoid division by zero
    df['cv'] = df['std_dev'] / df['vote_share_est']
    df['cv'] = df['cv'].replace([np.inf, -np.inf], np.nan)

    return df

# ==================== Plot 1: Global CV Distribution ====================


def plot_cv_distribution(df):
    plt.figure(figsize=(10, 6))

    # Filter reasonable range for visibility
    plot_data = df['cv'].dropna()
    # Remove extreme outliers for plot clarity
    plot_data = plot_data[plot_data < 2.0]

    sns.histplot(plot_data, bins=50, kde=True, color=COLOR_SAFE, alpha=0.6)

    mean_cv = df['cv'].mean()
    plt.axvline(mean_cv, color=COLOR_ELIM, linestyle='--',
                linewidth=2, label=f'Mean CV: {mean_cv:.4f}')

    plt.title('Distribution of Coefficient of Variation (Certainty) of Estimates',
              fontsize=16, fontweight='bold')
    plt.xlabel('Coefficient of Variation (CV)', fontsize=12)
    plt.ylabel('Frequency (Number of Prediction Instances)', fontsize=12)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, 'global_cv_distribution.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

# ==================== Plot 2: Global Scatter (Judge vs Fan) ====================


def plot_global_correlation(df):
    plt.figure(figsize=(12, 8))

    # Create normalized judge score for better comparison (0-1 scale is abstract, keep raw score for familiarity)
    # But coloring by Elimination status

    # Fix boolean for plot
    df['Status'] = df['is_eliminated'].apply(
        lambda x: 'Eliminated' if x else 'Safe')

    sns.scatterplot(
        data=df,
        x='judge_score',
        y='vote_share_est',
        hue='Status',
        palette={'Eliminated': COLOR_ELIM, 'Safe': COLOR_SAFE},
        alpha=0.5,
        s=60
    )

    plt.title('Global Correlation: Judge Scores vs. Estimated Fan Vote Share',
              fontsize=16, fontweight='bold')
    plt.xlabel('Total Judge Score', fontsize=12)
    plt.ylabel('Estimated Fan Vote Share', fontsize=12)
    plt.legend(title='Outcome', loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, 'ultimate_correlation.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

# ==================== Plot 3: Season 2 Jerry Rice Case Study ====================


def plot_jerry_rice_case(df):
    # Filter Season 2 and specific Names
    target_names = ['Jerry Rice', 'Drew Lachey']
    s2_df = df[(df['season'] == 2) & (
        df['celebrity_name'].isin(target_names))].copy()

    if s2_df.empty:
        print("Season 2 data not found in estimates.")
        return

    s2_df.sort_values(by='week', inplace=True)
    weeks = sorted(s2_df['week'].unique())
    bar_width = 0.35
    indices = np.arange(len(weeks))

    # Pivot data
    jerry_df = s2_df[s2_df['celebrity_name'] ==
                     'Jerry Rice'].set_index('week').reindex(weeks)
    drew_df = s2_df[s2_df['celebrity_name'] ==
                    'Drew Lachey'].set_index('week').reindex(weeks)

    # Setup Figure for Bar Chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Top: Judge Scores ---
    h1 = jerry_df['judge_score'].fillna(0)
    h2 = drew_df['judge_score'].fillna(0)

    rects1 = ax1.bar(indices - bar_width/2, h1, bar_width,
                     label='Jerry Rice', color=COLOR_JERRY, edgecolor='black', alpha=0.9)
    rects2 = ax1.bar(indices + bar_width/2, h2, bar_width,
                     label='Drew Lachey', color=COLOR_DREW, edgecolor='black', alpha=0.9)

    # Labels
    ax1.bar_label(rects1, padding=3, fmt='%.0f',
                  fontsize=10, fontweight='bold')
    ax1.bar_label(rects2, padding=3, fmt='%.0f',
                  fontsize=10, fontweight='bold')

    ax1.set_ylabel('Judge Score')
    ax1.set_title('Judge Scores Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 35)

    # --- Bottom: Estimated Vote Share ---
    h3 = jerry_df['vote_share_est'].fillna(0)
    h4 = drew_df['vote_share_est'].fillna(0)

    rects3 = ax2.bar(indices - bar_width/2, h3, bar_width,
                     label='Jerry Rice', color=COLOR_JERRY, edgecolor='black', alpha=0.9)
    rects4 = ax2.bar(indices + bar_width/2, h4, bar_width,
                     label='Drew Lachey', color=COLOR_DREW, edgecolor='black', alpha=0.9)

    # Labels (Percent)
    ax2.bar_label(rects3, padding=3, fmt='%.1%',
                  fontsize=10, fontweight='bold')
    ax2.bar_label(rects4, padding=3, fmt='%.1%',
                  fontsize=10, fontweight='bold')

    ax2.set_ylabel('Estimated Vote Share')
    ax2.set_title('Estimated Fan Vote Share Comparison',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(weeks)
    ax2.set_ylim(0, 0.6)

    fig.suptitle('Season 2: Jerry Rice (Runner-up) vs Drew Lachey (Winner)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, 'season2_jerry vs drew.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


# ==================== Plot 4: Season 2 Final Pie Chart ====================
def plot_season2_final_pie(df):
    # Filter Final Week of Season 2
    s2_df = df[df['season'] == 2]
    max_week = s2_df['week'].max()
    final_df = s2_df[s2_df['week'] == max_week].copy()

    if final_df.empty:
        return

    # Assuming we have 3 finalists usually
    names = final_df['celebrity_name'].tolist()

    # Judge Percentage (Need calculate from judge scores)
    total_judge = final_df['judge_score'].sum()
    judge_shares = (final_df['judge_score'] / total_judge).tolist()

    vote_shares = final_df['vote_share_est'].tolist()

    # Colors mapping
    # Ensure Jerry and Drew get their specific colors, others get generic
    c_map = []
    for n in names:
        if n == 'Jerry Rice':
            c_map.append(COLOR_JERRY)
        elif n == 'Drew Lachey':
            c_map.append(COLOR_DREW)
        else:
            c_map.append(COLOR_PURPLE)  # Stacy Keibler usually

    # Explode largest
    def get_explode(vals):
        max_idx = np.argmax(vals)
        return [0.05 if i == max_idx else 0.0 for i in range(len(vals))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Judge Pie
    wedges, texts, autotexts = axes[0].pie(
        judge_shares, labels=names, autopct='%1.1f%%', startangle=90,
        explode=get_explode(judge_shares), colors=c_map,
        wedgeprops=dict(edgecolor='w', width=1), textprops={'fontsize': 11}
    )
    axes[0].set_title('Judge Score Distribution (Finals)',
                      fontsize=14, fontweight='bold')

    # Vote Pie
    wedges, texts, autotexts = axes[1].pie(
        vote_shares, labels=names, autopct='%1.1f%%', startangle=90,
        explode=get_explode(vote_shares), colors=c_map,
        wedgeprops=dict(edgecolor='w', width=1), textprops={'fontsize': 11}
    )
    axes[1].set_title('Estimated Fan Vote Distribution (Finals)',
                      fontsize=14, fontweight='bold')

    fig.suptitle('Season 2 Finals: Technical Scores vs. Fan Popularity',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, 'season2_final_pie.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


# ==================== Main Runner ====================
if __name__ == '__main__':
    df = load_and_prep_data()

    print("Generating Plot 1: CV Distribution...")
    plot_cv_distribution(df)

    print("Generating Plot 2: Global Correlation...")
    plot_global_correlation(df)

    print("Generating Plot 3: Jerry Rice Bar Chart...")
    plot_jerry_rice_case(df)

    print("Generating Plot 4: Final Pie Chart...")
    plot_season2_final_pie(df)

    print("All plots generated successfully.")
