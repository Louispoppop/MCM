import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'Cleaned_data_with_votes.csv'
RESULTS_DIR = BASE_DIR / 'results'
GENERATED_FIGURES_DIR = BASE_DIR / 'figures' / 'generated'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 全局设置
# ==========================================
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 核心算法：自适应 Sigmoid
# ==========================================


def sigmoid_transformation(x, n_contestants, dynamic_k):
    """
    x: 原始得票率
    dynamic_k: 本周动态计算出的陡峭度
    """
    center = 1 / n_contestants
    # 核心公式
    transformed = 1 / (1 + np.exp(-dynamic_k * (x - center)))
    return transformed


def calculate_dynamic_k(fan_shares, base_k, sensitivity):
    """
    根据本周票数的离散程度计算 K 值
    逻辑：票数越接近(std越小)，K值越大(竞争越激烈)
    """
    std_val = np.std(fan_shares)

    # 防止 std 为 0 的极端情况
    if std_val < 0.001:
        std_val = 0.001

    k = base_k + (sensitivity / std_val)

    # 设置一个上限，防止 K 无限大导致数值计算溢出
    return min(k, 100)

# ==========================================
# 模拟主程序
# ==========================================


def task4_simulate_new_system(file_path):
    print("\n>>> 正在执行 Task 4：自适应新赛制 (Adaptive Logistic) 模拟...")
    df = pd.read_csv(file_path)

    comparison_results = []
    weekly_metrics = []

    grouped = df.groupby(['season', 'week'])

    for (season, week), g in tqdm(grouped):
        n = len(g)
        if n < 2:
            continue

        is_non_elim = (g['is_eliminated'].sum() == 0)
        names = g['celebrity_name'].values
        j_pcts = g['judge_percentage'].values
        j_ranks = g['week_rank'].values

        if 'final_est_share' in g.columns:
            fan_shares = g['final_est_share'].fillna(0).values
        else:
            fan_shares = g['voting_rate'].fillna(0).values

        fan_ranks = np.argsort(np.argsort(-fan_shares)) + 1

        # --- 1. 计算本周动态 K 值 ---
        week_k = calculate_dynamic_k(fan_shares, base_k=5, sensitivity=10.0)

        # --- 2. 规则 A: Rank Rule (原有规则 1) ---
        # 排名相加，值越小越好
        rank_sum = j_ranks + fan_ranks
        rank_rule_rank = np.argsort(np.argsort(rank_sum)) + 1

        # --- 3. 规则 B: Percent Rule (原有规则 2) ---
        # 百分比相加，值越大越好 (注意 argsort(-x))
        percent_score = j_pcts + fan_shares
        percent_rule_rank = np.argsort(np.argsort(-percent_score)) + 1

        # --- 4. 规则 C: Sigmoid Adaptive (新规则) ---
        fan_score_sigmoid = sigmoid_transformation(
            fan_shares, n, dynamic_k=week_k)

        # 归一化
        if np.sum(fan_score_sigmoid) > 0:
            fan_score_norm = fan_score_sigmoid / np.sum(fan_score_sigmoid)
        else:
            fan_score_norm = np.ones(n) / n

        new_score = 0.5 * j_pcts + 0.5 * fan_score_norm
        new_rank = np.argsort(np.argsort(-new_score)) + 1

        # --- 记录 ---
        for i, name in enumerate(names):
            comparison_results.append({
                'Season': season, 'Week': week, 'Celebrity': name,
                'Rank_Rule_Rank': rank_rule_rank[i],
                'Percent_Rule_Rank': percent_rule_rank[i],
                'New_Rank': new_rank[i],
                'Is_Non_Elimination': is_non_elim,
                'Raw_Share': fan_shares[i],
                'Dynamic_K': week_k
            })

        # --- 指标计算 ---
        # Fairness: 与评委排名的相关性
        fair_rank = spearmanr(rank_rule_rank, j_ranks)[0]
        fair_pct = spearmanr(percent_rule_rank, j_ranks)[0]
        fair_new = spearmanr(new_rank, j_ranks)[0]

        # Satisfaction: 与粉丝排名的相关性
        sat_rank = spearmanr(rank_rule_rank, fan_ranks)[0]
        sat_pct = spearmanr(percent_rule_rank, fan_ranks)[0]
        sat_new = spearmanr(new_rank, fan_ranks)[0]

        judge_top1_idx = np.argmin(j_ranks)
        risk_judge_rank = 1 if rank_rule_rank[judge_top1_idx] == n else 0
        risk_judge_pct = 1 if percent_rule_rank[judge_top1_idx] == n else 0
        risk_judge_new = 1 if new_rank[judge_top1_idx] == n else 0

        weekly_metrics.append({
            'Season': season, 'Week': week,
            'Fair_Rank': fair_rank, 'Fair_Pct': fair_pct, 'Fair_New': fair_new,
            'Sat_Rank': sat_rank, 'Sat_Pct': sat_pct, 'Sat_New': sat_new,
            'Risk_Judge_Rank': risk_judge_rank,
            'Risk_Judge_Pct': risk_judge_pct,
            'Risk_Judge_New': risk_judge_new,
            'Avg_K': week_k
        })

    res_df = pd.DataFrame(comparison_results)
    metrics_df = pd.DataFrame(weekly_metrics)

    return res_df, metrics_df

# ==========================================
# 可视化 1: 动态原理图 (保持不变)
# ==========================================


def plot_dynamic_mechanism():
    output_path = GENERATED_FIGURES_DIR / 'task4_mechanism_adaptive_generated.png'
    print(f" -> 正在绘制自适应原理图 ({output_path.name})...")

    x = np.linspace(0, 0.2, 400)
    n = 10
    center = 1/n

    k_tight = 80
    k_loose = 20

    y_linear = x

    # --- 场景 A: 激烈竞争 (High K) ---
    s_target_tight = sigmoid_transformation(x, n, k_tight)
    s_other_tight = sigmoid_transformation((1-x)/(n-1), n, k_tight)
    s_total_tight = s_target_tight + (n-1)*s_other_tight
    y_tight = s_target_tight / s_total_tight

    # --- 场景 B: 悬殊竞争 (Low K) ---
    s_target_loose = sigmoid_transformation(x, n, k_loose)
    s_other_loose = sigmoid_transformation((1-x)/(n-1), n, k_loose)
    s_total_loose = s_target_loose + (n-1)*s_other_loose
    y_loose = s_target_loose / s_total_loose

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_linear, 'k--', linewidth=2, alpha=0.4,
             label='Original: Linear (Percent Rule)')
    plt.plot(x, y_tight, color='#3e7cd8', linestyle='-', linewidth=3,
             label='Scenario A: Tight Race (Higher K=80)')
    plt.plot(x, y_loose, color='#ee4d7a', linestyle='-',
             linewidth=3, label='Scenario B: Blowout (Lower K=20)')

    plt.axvline(center, color='gray', linestyle=':', alpha=0.6)
    plt.text(center, 0.0, ' Avg (10%)', color='gray',
             fontsize=10, ha='center', va='bottom')

    plt.annotate('Steeper Slope:\nIntense Competition',
                 xy=(0.06, 0.15), xytext=(0.06, 0.15),
                 arrowprops=dict(facecolor='#3e7cd8', arrowstyle='->'), color='#3e7cd8', fontweight='bold')

    plt.annotate('Softer Slope:\nStandard Suppression',
                 xy=(0.15, 0.10), xytext=(0.15, 0.10),
                 arrowprops=dict(facecolor='#ee4d7a', arrowstyle='->'), color='#ee4d7a', fontweight='bold')
    # Choose x = 0.15 (15% Vote) for comparison
    x1_ref = 0.18

    # Calculate y at x=0.15 for Tight
    s_t1_ref = sigmoid_transformation(np.array([x1_ref]), n, k_tight)[0]
    s_o1_ref = sigmoid_transformation(
        np.array([(1-x1_ref)/(n-1)]), n, k_tight)[0]
    y_tight_ref = s_t1_ref / (s_t1_ref + (n-1)*s_o1_ref)

    s_t2_ref = sigmoid_transformation(np.array([x1_ref]), n, k_loose)[0]
    s_o2_ref = sigmoid_transformation(
        np.array([(1-x1_ref)/(n-1)]), n, k_loose)[0]
    y_loose_ref = s_t2_ref / (s_t2_ref + (n-1)*s_o2_ref)
    # Draw reference line
    plt.plot([x1_ref, x1_ref], [0, y_tight_ref],
             color='gray', linestyle=':', alpha=0.5)
    plt.plot([x1_ref, x1_ref], [0, y_loose_ref],
             color='gray', linestyle=':', alpha=0.5)

    # Annotate Tight vs Linear
    diff_tight = y_tight_ref - y_loose_ref
    # Draw bracket/line
    plt.plot([x1_ref, x1_ref], [y_loose_ref, y_tight_ref],
             color='#9C86DA', linewidth=2)
    plt.scatter([x1_ref, x1_ref], [y_loose_ref, y_tight_ref],
                color='#9C86DA', s=15)
    plt.text(x1_ref + 0.01, (y_loose_ref + y_tight_ref)/2, f'+{diff_tight:.2f}\nvs Loose',
             color='#9C86DA', fontsize=9, fontweight='bold', va='center')

    x2_ref = 0.02
    s_t1_ref2 = sigmoid_transformation(np.array([x2_ref]), n, k_tight)[0]
    s_o1_ref2 = sigmoid_transformation(
        np.array([(1-x2_ref)/(n-1)]), n, k_tight)[0]
    y_tight_ref2 = s_t1_ref2 / (s_t1_ref2 + (n-1)*s_o1_ref2)
    s_t2_ref2 = sigmoid_transformation(np.array([x2_ref]), n, k_loose)[0]
    s_o2_ref2 = sigmoid_transformation(
        np.array([(1-x2_ref)/(n-1)]), n, k_loose)[0]
    y_loose_ref2 = s_t2_ref2 / (s_t2_ref2 + (n-1)*s_o2_ref2)
    diff_loose = y_loose_ref2 - y_tight_ref2
    plt.plot([x2_ref, x2_ref], [y_tight_ref2, y_loose_ref2],
             color='#9C86DA', linewidth=2)
    plt.scatter([x2_ref, x2_ref], [y_tight_ref2,
                y_loose_ref2], color='#9C86DA', s=15)
    plt.text(x2_ref + 0.02, (y_tight_ref2 + y_loose_ref2)/2, f'+{diff_loose:.2f}\nvs Tight',
             color='#9C86DA', fontsize=9, fontweight='bold', va='center', ha='right')

    plt.title(
        'Mechanism of Adaptive Meritocracy: Dynamic K-Factor (Normalized)', fontsize=15)
    plt.xlabel('Raw Fan Vote Share', fontsize=12)
    plt.ylabel('Effective Score (Normalized)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    plt.savefig(output_path, dpi=300)
    plt.close()

# ==========================================
# 可视化 2: 案例影响 (Bobby Bones) (保持不变)
# ==========================================


def plot_impact_analysis_refined(res_df, target_celeb="Bobby Bones"):
    safe_name = target_celeb.replace(' ', '_')
    output_path = GENERATED_FIGURES_DIR / \
        f"task4_impact_{safe_name.lower()}_generated.png"
    print(f" -> Generating Impact Analysis Plot ({output_path.name})...")
    celeb_df = res_df[res_df['Celebrity'] ==
                      target_celeb].sort_values(['Season', 'Week'])

    # --- NEW: Remove Week 5 for Bobby Bones ---
    if target_celeb == "Bobby Bones":
        celeb_df = celeb_df[celeb_df['Week'] != 5]
    # ------------------------------------------

    if celeb_df.empty:
        return

    weeks = [f"W{w}" for w in celeb_df['Week']]
    old_ranks = celeb_df['Percent_Rule_Rank'].values
    new_ranks = celeb_df['New_Rank'].values
    is_non_elim = celeb_df['Is_Non_Elimination'].values

    plt.figure(figsize=(12, 6))
    plt.plot(weeks, old_ranks, '--', color='#9C86DA', alpha=0.5)
    plt.plot(weeks, new_ranks, '-', color='#3e7cd8', alpha=0.5)

    mask_norm = ~is_non_elim
    if np.any(mask_norm):
        plt.scatter(np.array(weeks)[mask_norm], old_ranks[mask_norm],
                    color='#9C86DA', s=80, label='Original (Percent)')
        plt.scatter(np.array(weeks)[mask_norm], new_ranks[mask_norm],
                    color='#3e7cd8', marker='s', s=80, label='Adaptive System')

    mask_ne = is_non_elim
    if np.any(mask_ne):
        plt.scatter(np.array(weeks)[mask_ne], old_ranks[mask_ne],
                    facecolors='none', edgecolors='#9C86DA', s=150, linestyle=':')
        plt.scatter(np.array(weeks)[mask_ne], new_ranks[mask_ne], facecolors='none',
                    edgecolors='#3e7cd8', marker='s', s=150, linestyle=':')

    for i, (o, n, ne) in enumerate(zip(old_ranks, new_ranks, is_non_elim)):
        if n > o:
            alpha = 0.3 if ne else 0.8
            plt.arrow(i, o, 0, n-o-0.2, head_width=0.1,
                      fc='#ee4d7a', ec='#ee4d7a', alpha=alpha)
            txt = f'( -{n-o} )' if ne else f'-{n-o}'
            plt.text(i+0.1, (o+n)/2, txt, color='#ee4d7a', va='center',
                     fontsize=20, fontdict={'weight': 'bold', 'alpha': alpha})

    plt.gca().invert_yaxis()
    plt.title(
        f'Impact Analysis: {target_celeb} (Adaptive System)', fontsize=15)
    plt.ylabel('Weekly Rank', fontsize=12)
    plt.xlabel('Week', fontsize=12)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300)
    plt.close()

# ==========================================
# 可视化 3: 客观指标对比 (三方对比: Rank vs New vs Percent) - 增强版标注
# ==========================================


def visualize_metrics_comparison_final_three_way(metrics_df):
    output_path = GENERATED_FIGURES_DIR / \
        'task4_metrics_final_threeway_generated.png'
    print(f" -> 正在生成三方对比评分图 ({output_path.name})...")

    # 1. 计算均值
    avg_fair_rank = metrics_df['Fair_Rank'].mean()
    avg_fair_pct = metrics_df['Fair_Pct'].mean()
    avg_fair_new = metrics_df['Fair_New'].mean()

    avg_sat_rank = metrics_df['Sat_Rank'].mean()
    avg_sat_pct = metrics_df['Sat_Pct'].mean()
    avg_sat_new = metrics_df['Sat_New'].mean()

    print(f"\n[三方评分结果]")
    print(
        f"Fairness: Rank={avg_fair_rank:.4f}, New={avg_fair_new:.4f}, Pct={avg_fair_pct:.4f}")
    print(
        f"Satisfaction: Rank={avg_sat_rank:.4f}, New={avg_sat_new:.4f}, Pct={avg_sat_pct:.4f}")

    # 2. 准备绘图数据
    labels = ['Fairness\n(Judge Correlation)',
              'Satisfaction\n(Fan Correlation)']

    rank_vals = [avg_fair_rank, avg_sat_rank]
    new_vals = [avg_fair_new, avg_sat_new]
    pct_vals = [avg_fair_pct, avg_sat_pct]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # 绘制三个柱子
    rects1 = ax.bar(x - width, rank_vals, width,
                    label='Rule 1: Rank System', color='#3e7cd8', alpha=0.8)
    rects2 = ax.bar(x, new_vals, width,
                    label='Proposed: Adaptive System', color='#ee4d7a', alpha=0.9)
    rects3 = ax.bar(x + width, pct_vals, width,
                    label='Rule 2: Percent System', color='#9C86DA', alpha=0.8)

    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title(
        'Comprehensive Comparison: Rank vs Proposed vs Percent', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower center', ncol=3)
    ax.set_ylim(0, 1.4)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}', xy=(rect.get_x()+rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # --- Helper: Annotate ---
    def annotate_bracket(x_start, x_end, y_h, diff, pct_diff, color, label_suffix):
        ax.plot([x_start, x_start, x_end, x_end], [
                y_h-0.02, y_h, y_h, y_h-0.02], color=color, lw=1.5)
        txt = f'{diff:+.3f}\n({pct_diff:+.1%})\n{label_suffix}'
        ax.text((x_start + x_end) / 2, y_h + 0.02, txt, ha='center',
                va='bottom', color=color, fontweight='bold', fontsize=9)

    # 1. Fairness: Rank vs New
    diff_fair_rank = avg_fair_new - avg_fair_rank
    pct_fair_rank = diff_fair_rank / avg_fair_rank
    color_fair_rank = '#d62828' if diff_fair_rank >= 0 else 'green'
    y_h_fair_1 = max(avg_fair_rank, avg_fair_new) + 0.08
    annotate_bracket(x[0]-width, x[0], y_h_fair_1, diff_fair_rank,
                     pct_fair_rank, color_fair_rank, "vs Rank")

    # 2. Fairness: New vs Percent
    diff_fair_pct = avg_fair_new - avg_fair_pct
    pct_fair_pct = diff_fair_pct / avg_fair_pct
    color_fair_pct = '#d62828' if diff_fair_pct >= 0 else 'green'
    y_h_fair_2 = max(avg_fair_new, avg_fair_pct) + 0.08
    annotate_bracket(x[0], x[0]+width, y_h_fair_2,
                     diff_fair_pct, pct_fair_pct, color_fair_pct, "vs Pct")

    # 3. Satisfaction: Rank vs New
    diff_sat_rank = avg_sat_new - avg_sat_rank
    pct_sat_rank = diff_sat_rank / avg_sat_rank
    color_sat_rank = '#d62828' if diff_sat_rank >= 0 else 'green'
    y_h_sat_1 = max(avg_sat_rank, avg_sat_new) + 0.08
    annotate_bracket(x[1]-width, x[1], y_h_sat_1, diff_sat_rank,
                     pct_sat_rank, color_sat_rank, "vs Rank")

    # 4. Satisfaction: New vs Percent
    diff_sat_pct = avg_sat_new - avg_sat_pct
    pct_sat_pct = diff_sat_pct / avg_sat_pct
    color_sat_pct = '#d62828' if diff_sat_pct >= 0 else 'green'
    # Stagger to avoid overlap with previous bracket
    y_h_sat_2 = max(avg_sat_new, avg_sat_pct) + 0.08
    annotate_bracket(x[1], x[1]+width, y_h_sat_2, diff_sat_pct,
                     pct_sat_pct, color_sat_pct, "vs Pct")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    res_df, metrics_df = task4_simulate_new_system(DATA_FILE)
    res_df.to_csv(
        RESULTS_DIR / 'task4_simulation_results_generated.csv', index=False)
    plot_dynamic_mechanism()
    plot_impact_analysis_refined(res_df, "Bobby Bones")
    visualize_metrics_comparison_final_three_way(metrics_df)
    print("\n>>> Task 4 Completed.")
