"""
Script to redraw the 'Average CV by Season' plot with enhanced styling and specific range constraints.
Reads input from 'task1_estimates_complete.csv'.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, "task1_estimates_complete.csv")
    output_path = os.path.join(base_dir, "analysis_cv_by_season_enhanced.png")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # Load data
    df = pd.read_csv(input_path)

    # Calculate average CV per season
    season_cv = df.groupby("season")["cv"].mean().reset_index()

    # Setup style
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(14, 7))

    # Create Bar Plot
    # Using a gradient-like palette or a distinct clean color
    ax = sns.barplot(
        data=season_cv,
        x="season",
        y="cv",
        color="#5DADE2",  # A nice soft blue
        edgecolor="white",
        linewidth=0.5
    )

    # Set Y-axis limits
    plt.ylim(0, 1.0)

    # Add Threshold Lines
    # High Certainty Threshold (< 0.1)
    plt.axhline(y=0.3, color="#206358FF", linestyle='--',
                linewidth=2, label='High Certainty Threshold (0.3)')
    # Low Certainty Threshold (> 0.3)
    plt.axhline(y=0.6, color="#C66C2C", linestyle='--',
                linewidth=2, label='Low Certainty Threshold (0.6)')

    # Zone: High Certainty
    plt.axhspan(0, 0.3, color="#2ECC50", alpha=0.1)
    # Zone: Medium Certainty
    plt.axhspan(0.3, 0.6, color='#F1C40F', alpha=0.1)
    # Zone: Low Certainty
    plt.axhspan(0.6, 1.0, color='#E74C3C', alpha=0.1)
    # Labels and Title
    plt.title("Average Fan Vote Uncertainty (CV) by Season",
              fontsize=18, pad=20, weight='bold')
    plt.xlabel("Season", fontsize=14, labelpad=10)
    plt.ylabel("Average Coefficient of Variation (CV)",
               fontsize=14, labelpad=10)

    # Ticks formatting
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=12)

    # Legend
    plt.legend(loc='upper right', frameon=True,
               framealpha=0.9, fancybox=True, shadow=True)

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Enhanced plot saved to: {output_path}")


if __name__ == "__main__":
    main()
