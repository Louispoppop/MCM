import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm

# 全局设置
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 核心函数
# ==========================================
def sigmoid_transformation(x, n_contestants, dynamic_k):
    center = 1 / n_contestants
    transformed = 1 / (1 + np.exp(-dynamic_k * (x - center)))
    return transformed

def calculate_dynamic_k(fan_shares, base_k, sensitivity):
    std_val = np.std(fan_shares)
    if std_val < 0.001: std_val = 0.001
    k = base_k + (sensitivity / std_val)
    return min(k, 100) 

def run_simulation_for_params(df, base_k, sensitivity):
    fairness_scores = []
    satisfaction_scores = []
    
    grouped = df.groupby(['season', 'week'])
    
    for _, g in grouped:
        n = len(g)
        if n < 2: continue
        
        j_pcts = g['judge_percentage'].values
        j_ranks = g['week_rank'].values
        
        if 'final_est_share' in g.columns:
            fan_shares = g['final_est_share'].fillna(0).values
        else:
            fan_shares = g['voting_rate'].fillna(0).values
        
        fan_ranks = np.argsort(np.argsort(-fan_shares)) + 1
        
        # --- 新规则计算 ---
        week_k = calculate_dynamic_k(fan_shares, base_k, sensitivity)
        fan_score_sigmoid = sigmoid_transformation(fan_shares, n, dynamic_k=week_k)
        
        if np.sum(fan_score_sigmoid) > 0:
            fan_score_norm = fan_score_sigmoid / np.sum(fan_score_sigmoid)
        else:
            fan_score_norm = np.ones(n) / n
            
        new_score = 0.5 * j_pcts + 0.5 * fan_score_norm
        new_rank = np.argsort(np.argsort(-new_score)) + 1
        
        # 记录指标
        f_score, _ = spearmanr(new_rank, j_ranks)
        s_score, _ = spearmanr(new_rank, fan_ranks)
        
        if not np.isnan(f_score): fairness_scores.append(f_score)
        if not np.isnan(s_score): satisfaction_scores.append(s_score)
        
    return np.mean(fairness_scores), np.mean(satisfaction_scores)

# ==========================================
# 敏感度分析主程序
# ==========================================
def run_sensitivity_analysis(file_path):
    print(">>> 开始参数敏感度分析 (Grid Search)...")
    df = pd.read_csv(file_path)
    
    # 【修改点1】移除 0，聚焦有效参数范围
    base_k_list = [5, 10, 15, 20, 25] 
    sensitivity_list = [5, 10, 20, 30, 50, 80]
    
    results = []
    
    total_iters = len(base_k_list) * len(sensitivity_list)
    pbar = tqdm(total=total_iters)
    
    for bk in base_k_list:
        for sens in sensitivity_list:
            avg_fair, avg_sat = run_simulation_for_params(df, bk, sens)
            
            # 综合得分 (简单平均)
            composite = 0.5 * avg_fair + 0.5 * avg_sat
            
            results.append({
                'Base_K': bk,
                'Sensitivity': sens,
                'Fairness': avg_fair,
                'Satisfaction': avg_sat,
                'Composite': composite
            })
            pbar.update(1)
            
    pbar.close()
    
    res_df = pd.DataFrame(results)
    return res_df

# ==========================================
# 可视化函数 (增强对比度版)
# ==========================================
def plot_sensitivity_heatmaps(res_df):
    print(" -> 正在绘制高对比度热力图...")
    
    pivot_fair = res_df.pivot(index='Base_K', columns='Sensitivity', values='Fairness')
    pivot_sat = res_df.pivot(index='Base_K', columns='Sensitivity', values='Satisfaction')
    pivot_comp = res_df.pivot(index='Base_K', columns='Sensitivity', values='Composite')
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 辅助函数：绘制单个热力图
    def draw_heatmap(data, ax, title, cmap):
        # 【修改点2】动态计算 vmin 和 vmax，强制拉伸颜色范围，增强对比
        vmin = data.min().min()
        vmax = data.max().max()
        
        sns.heatmap(data, annot=True, fmt=".3f", cmap=cmap, ax=ax, 
                    vmin=vmin, vmax=vmax,  # 关键：锁定极值范围
                    cbar_kws={'label': 'Score'})
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis() # 习惯上y轴从小到大向上
    
    # 1. Fairness Heatmap
    draw_heatmap(pivot_fair, axes[0], 'Metric: Fairness (Judge Correlation)', "YlGnBu")
    axes[0].set_ylabel('Base K (Steepness Baseline)')
    
    # 2. Satisfaction Heatmap
    draw_heatmap(pivot_sat, axes[1], 'Metric: Satisfaction (Fan Correlation)', "YlOrRd")
    axes[1].set_ylabel('')
    
    # 3. Composite Score Heatmap
    draw_heatmap(pivot_comp, axes[2], 'Composite Score (Balanced)', "RdYlGn")
    axes[2].set_ylabel('')
    
    # 高亮标记出 (5, 10) 的位置
    try:
        # 获取坐标位置
        bk_vals = sorted(res_df['Base_K'].unique())
        sens_vals = sorted(res_df['Sensitivity'].unique())
        
        if 5 in bk_vals and 10 in sens_vals:
            row_idx = bk_vals.index(5) # Base K 对应行
            col_idx = sens_vals.index(10) # Sensitivity 对应列
            
            # 在三个图上都画框
            for ax in axes:
                # 注意：heatmap的坐标系原点在左上角，但在sns.heatmap中如果未invert_yaxis
                # index 0 (Base_K=5) 在最上面。
                # 如果做了 invert_yaxis (Base_K=5在最下面)，需要注意坐标对应。
                # Seaborn heatmap 默认 index 0 在最上面。
                # 为了保险，我们直接用 matplotlib patch，根据单元格索引画框
                
                # 由于前面用了 ax.invert_yaxis()，Base_K=5 (index 0) 会在最下面
                # 实际上 seaborn 默认是 0 在上，invert 后 0 在下。
                # Base_K 列表是 sorted 的: [5, 10, 15, 20, 25]
                # 5 是第一个元素，index=0。
                # 如果 invert_yaxis 了，index 0 在坐标轴的 0 位置（底部）。
                
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='blue', lw=3, clip_on=False))
    except ValueError:
        pass 
        
    plt.tight_layout()
    plt.savefig('Task4_Parameter_Sensitivity.png', dpi=300)
    print(" -> 图表已保存为 Task4_Parameter_Sensitivity.png")

if __name__ == "__main__":
    file_path = 'Cleaned_data_with_votes.csv'
    res_df = run_sensitivity_analysis(file_path)
    plot_sensitivity_heatmaps(res_df)
    
    selected = res_df[(res_df['Base_K']==5) & (res_df['Sensitivity']==10)]
    print("\n[Selected Parameters Performance]")
    print(selected.to_string(index=False))