import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "sensitivity_results.csv")
OUTPUT_HEATMAP = os.path.join(BASE_DIR, "sensitivity_accuracy.png")
OUTPUT_CV = os.path.join(BASE_DIR, "sensitivity_cv.png")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)

    # Prepare Data for 3D Plotting
    lambdas = sorted(df['Lambda'].unique())  # Y axis
    kappas = sorted(df['Kappa'].unique())   # X axis
    X, Y = np.meshgrid(kappas, lambdas)

    pivot_acc = df.pivot(index="Lambda", columns="Kappa", values="Accuracy")
    pivot_cv = df.pivot(index="Lambda", columns="Kappa", values="AvgCV")
    Z_acc = pivot_acc.values
    Z_cv = pivot_cv.values

    # Plot 1: Accuracy 3D Surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_acc, cmap=cm.viridis, vmin=0.95, vmax=1.0,
                           linewidth=0.1, edgecolors='k', alpha=0.9, antialiased=True)
    ax.set_xlabel('Kappa (Variance)', labelpad=10)
    ax.set_ylabel('Lambda (Performance Weight)', labelpad=10)
    ax.set_zlabel('Reproduction Rate', labelpad=10)
    ax.set_zlim(0.95, 1.0)
    ax.set_title("Model Robustness: Historical Reproduction Rate", pad=20)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Reproduction Rate')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP, dpi=300)
    print(f"Saved {OUTPUT_HEATMAP}")
    plt.close()

    # Plot 2: CV 3D Surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_cv, cmap=cm.magma_r, linewidth=0.1,
                           edgecolors='k', alpha=0.9, antialiased=True)
    ax.set_xlabel('Kappa (Variance)', labelpad=10)
    ax.set_ylabel('Lambda (Performance Weight)', labelpad=10)
    ax.set_zlabel('Avg CV (Uncertainty)', labelpad=10)
    ax.set_title("Model Uncertainty: Average CV", pad=20)
    ax.invert_xaxis()
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Coefficient of Variation')
    ax.view_init(elev=30, azim=-120)
    plt.tight_layout()
    plt.savefig(OUTPUT_CV, dpi=300)
    print(f"Saved {OUTPUT_CV}")
    plt.close()


if __name__ == "__main__":
    main()
