import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from tqdm import tqdm

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------
# 分任务 1：规则对比与粉丝偏好分析 (含可视化)
# ----------------------------------------------------------
def task2_subtask1_rules_comparison(file_path):
    print(">>> 正在执行分任务 1：规则对比与粉丝偏好分析...")
    df = pd.read_csv(file_path)
    
    results = []
    
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), g in tqdm(grouped):
        n = len(g)
        if n < 2: continue 
        
        j_ranks = g['week_rank'].values
        j_pcts = g['judge_percentage'].values
        
        if 'final_est_share' in g.columns:
            fan_shares = g['final_est_share'].fillna(0).values
        else:
            fan_shares = g['voting_rate'].fillna(0).values
            
        fan_ranks = np.argsort(np.argsort(-fan_shares)) + 1
        
        # Rank Rule
        total_rank_A = j_ranks + fan_ranks
        final_rank_A = np.argsort(np.argsort(total_rank_A)) + 1
        
        # Percent Rule
        total_pct_B = j_pcts + fan_shares
        final_rank_B = np.argsort(np.argsort(-total_pct_B)) + 1
        
        # Metrics
        dist_A = np.sum(np.abs(final_rank_A - fan_ranks))
        dist_B = np.sum(np.abs(final_rank_B - fan_ranks))
        
        norm_dist_A = dist_A / n
        norm_dist_B = dist_B / n
        
        winner = 'Tie'
        if norm_dist_A < norm_dist_B: winner = 'Rank-based'
        elif norm_dist_B < norm_dist_A: winner = 'Percent-based'
            
        results.append({
            'season': season,
            'week': week,
            'dist_Rank': norm_dist_A,
            'dist_Percent': norm_dist_B,
            'Winner': winner
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n[分任务 1 结论] 哪种规则更偏向粉丝？")
    print(f" -> Rank 规则平均偏离度: {res_df['dist_Rank'].mean():.4f}")
    print(f" -> Percent 规则平均偏离度: {res_df['dist_Percent'].mean():.4f} (数值越小越好)")
    
    # --- 可视化 ---
    print(" -> 正在生成分任务 1 可视化图表...")
    
    # 1. 箱线图
    plt.figure(figsize=(10, 6))
    melted_df = res_df.melt(value_vars=['dist_Rank', 'dist_Percent'], var_name='Rule', value_name='FDI')
    melted_df['Rule'] = melted_df['Rule'].replace({'dist_Rank': 'Rank-based', 'dist_Percent': 'Percent-based'})
    
    sns.boxplot(x='Rule', y='FDI', data=melted_df, width=0.5, palette="Set2", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})
    
    mean_rank = res_df['dist_Rank'].mean()
    mean_pct = res_df['dist_Percent'].mean()
    plt.text(0, mean_rank, f'Mean: {mean_rank:.4f}', ha='center', va='bottom', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.text(1, mean_pct, f'Mean: {mean_pct:.4f}', ha='center', va='bottom', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title('Distribution of Fan Deviation Index (FDI)', fontsize=14)
    plt.ylabel('Normalized Deviation (Lower is Better)')
    plt.savefig('Task2_Subtask1_Boxplot.png', dpi=300)
    plt.close()
    
    # 2. 柱状图 (颜色已修正)
    plt.figure(figsize=(8, 6))
    win_counts = res_df['Winner'].value_counts()
    
    # === [颜色修正] ===
    custom_colors = ["#3e7cd8", "#ee4d7a", '#9C86DA']
    colors_to_use = custom_colors[:len(win_counts)]
    
    bars = plt.bar(win_counts.index, win_counts.values, color=colors_to_use)
    
    plt.title('Winning Counts: Which Rule follows Fan Preference more often?')
    plt.ylabel('Number of Weeks')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig('Task2_Subtask1_BarChart.png', dpi=300)
    plt.close()
    
    return res_df

# ----------------------------------------------------------
# 分任务 2：争议选手深度案例分析 (忽略非淘汰周)
# ----------------------------------------------------------
def task2_subtask2_controversial_analysis(file_path, target_celebs):
    print(f"\n>>> 正在执行分任务 2：争议选手深度分析 {target_celebs}...")
    
    df = pd.read_csv(file_path)
    detailed_results = []
    
    for celeb in target_celebs:
        seasons_involved = df[df['celebrity_name'] == celeb]['season'].unique()
        seasons_involved.sort()
        
        if len(seasons_involved) == 0:
            print(f"警告: 数据中找不到选手 {celeb}")
            continue
            
        print(f" -> Found {celeb} in Season(s): {seasons_involved}")
        
        for season in seasons_involved:
            celeb_rows = df[
                (df['celebrity_name'] == celeb) & 
                (df['season'] == season)
            ].sort_values('week')
            
            for _, row in celeb_rows.iterrows():
                week = row['week']
                g = df[(df['season'] == season) & (df['week'] == week)]
                n = len(g)
                if n < 3: continue 
                
                # 忽略非淘汰周
                if g['is_eliminated'].sum() == 0:
                    continue
                
                names = g['celebrity_name'].values
                j_ranks = g['week_rank'].values           
                j_pcts = g['judge_percentage'].values     
                
                if 'final_est_share' in g.columns:
                    fan_shares = g['final_est_share'].fillna(0).values
                else:
                    fan_shares = g['voting_rate'].fillna(0).values
                    
                fan_ranks = np.argsort(np.argsort(-fan_shares)) + 1
                
                try:
                    target_idx = np.where(names == celeb)[0][0]
                except IndexError:
                    continue
                    
                # 模拟 1: Rank
                total_rank = j_ranks + fan_ranks
                max_rank_val = np.max(total_rank)
                elim_indices_rank = np.where(total_rank == max_rank_val)[0]
                res_rank = "ELIMINATED" if target_idx in elim_indices_rank else "Safe"
                    
                # 模拟 2: Rank + Save
                sorted_indices = np.argsort(total_rank) 
                bottom2_indices = sorted_indices[-2:] 
                res_rank_save = "Safe"
                if target_idx in bottom2_indices:
                    opponent_idx = bottom2_indices[0] if bottom2_indices[1] == target_idx else bottom2_indices[1]
                    if j_ranks[target_idx] < j_ranks[opponent_idx]:
                        res_rank_save = "SAVED (Judge)"  
                    elif j_ranks[target_idx] > j_ranks[opponent_idx]:
                        res_rank_save = "ELIMINATED"     
                    else:
                        res_rank_save = "ELIMINATED (Tie)" 
                else:
                    res_rank_save = "Safe"

                # 模拟 3: Percent
                total_pct = j_pcts + fan_shares
                min_pct_val = np.min(total_pct)
                elim_indices_pct = np.where(total_pct == min_pct_val)[0]
                res_pct = "ELIMINATED" if target_idx in elim_indices_pct else "Safe"
                    
                # 模拟 4: Percent + Save
                sorted_indices_pct = np.argsort(total_pct)
                bottom2_indices_pct = sorted_indices_pct[:2]
                res_pct_save = "Safe"
                if target_idx in bottom2_indices_pct:
                    opponent_idx = bottom2_indices_pct[0] if bottom2_indices_pct[1] == target_idx else bottom2_indices_pct[1]
                    if j_pcts[target_idx] > j_pcts[opponent_idx]:
                        res_pct_save = "SAVED (Judge)"
                    elif j_pcts[target_idx] < j_pcts[opponent_idx]:
                        res_pct_save = "ELIMINATED"
                    else:
                        res_pct_save = "ELIMINATED (Tie)"
                else:
                    res_pct_save = "Safe"

                actual_status = "ELIMINATED" if row['is_eliminated'] == 1 else "Safe"

                detailed_results.append({
                    'Celebrity': celeb, 'Season': season, 'Week': week,
                    'Judge_Rank': j_ranks[target_idx], 'Fan_Rank': fan_ranks[target_idx],
                    'Fan_Share': fan_shares[target_idx], 'ACTUAL': actual_status,
                    'Sim_Rank': res_rank, 'Sim_Rank_Save': res_rank_save,
                    'Sim_Percent': res_pct, 'Sim_Percent_Save': res_pct_save
                })

    res_df = pd.DataFrame(detailed_results)
    
    def check_reversal(row, col_sim):
        if row['ACTUAL'] == 'Safe' and 'ELIMINATED' in row[col_sim]: return 'DANGER' 
        if row['ACTUAL'] == 'ELIMINATED' and 'ELIMINATED' not in row[col_sim]: return 'SURVIVED'
        return ''

    for col in ['Sim_Rank', 'Sim_Rank_Save', 'Sim_Percent', 'Sim_Percent_Save']:
        res_df[f'Diff_{col}'] = res_df.apply(lambda r: check_reversal(r, col), axis=1)

    return res_df

# ----------------------------------------------------------
# 分任务 2 可视化：生存时间轴 (已修正：分赛季独立绘图)
# ----------------------------------------------------------
def task2_subtask2_visualize_timelines(csv_path='Task2_Controversial_Analysis.csv'):
    print(f"\n>>> 正在生成分任务 2 的生存时间轴可视化 (逻辑修正版)...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("错误: 找不到结果文件")
        return

    # 状态映射: 0:Safe, 1:Saved, 2:Elim, 3:Ghost
    def map_status(val):
        val = str(val).upper()
        if 'SAVED' in val: return 1
        if 'ELIMINATED' in val: return 2
        return 0 

    # 颜色: 绿, 金, 红, 灰
    cmap = ListedColormap(['#66b032', '#f0c808', '#d62828', '#e0e0e0']) 
    
    rules_order = ['ACTUAL', 'Sim_Rank', 'Sim_Rank_Save', 'Sim_Percent', 'Sim_Percent_Save']
    rules_labels = ['Actual History', 'Rank Rule', 'Rank + Save', 'Percent Rule', 'Percent + Save']
    
    # === 修改点：按 (Celebrity, Season) 分组绘图 ===
    # 这样可以确保同一个明星如果参加了多个赛季，会生成两张独立的图
    grouped = df.groupby(['Celebrity', 'Season'])
    
    for (celeb, season), celeb_data in grouped:
        # 按周排序
        celeb_data = celeb_data.sort_values(['Week'])
        
        if celeb_data.empty: continue
            
        # X轴标签：只显示 W{week}，因为Season已经分开了
        weeks_labels = [f"W{row.Week}" for _, row in celeb_data.iterrows()]
        
        plot_matrix = []
        for rule in rules_order:
            raw_states = celeb_data[rule].apply(map_status).values
            
            # --- 幽灵截断逻辑 ---
            processed_states = []
            is_dead = False
            
            for i, state in enumerate(raw_states):
                # 决赛周判断
                is_final_week = (i == len(raw_states) - 1)
                
                if rule == 'ACTUAL': 
                    processed_states.append(state)
                    continue
                
                if is_dead:
                    processed_states.append(3) # 变灰
                else:
                    processed_states.append(state)
                    if state == 2 and not is_final_week:
                        is_dead = True
            
            plot_matrix.append(processed_states)
            
        plot_matrix = np.array(plot_matrix)
        
        # 绘图
        plt.figure(figsize=(max(8, len(weeks_labels)*0.9), 5.5))
        ax = sns.heatmap(plot_matrix, cmap=cmap, linewidths=1, linecolor='white', 
                         cbar=False, vmin=0, vmax=3, annot=False)
        
        ax.set_xticklabels(weeks_labels, rotation=0, fontsize=10)
        ax.set_yticklabels(rules_labels, rotation=0, fontsize=11, fontweight='bold')
        
        legend_handles = [
            Patch(facecolor='#66b032', edgecolor='white', label='Safe / Finalist'),
            Patch(facecolor='#f0c808', edgecolor='white', label='Saved by Judges'),
            Patch(facecolor='#d62828', edgecolor='white', label='Eliminated'),
            Patch(facecolor='#e0e0e0', edgecolor='gray', label='Absent')
        ]
        plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=4, frameon=False, fontsize=11)
        
        # 标题加入赛季信息
        plt.title(f'Counterfactual Survival Timeline: {celeb} (Season {season})', fontsize=16, pad=20)
        plt.tight_layout()
        
        # 文件名加入赛季后缀
        safe_name = celeb.replace(" ", "_")
        filename = f'Task2_Subtask2_Timeline_{safe_name}_S{season}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> 已生成: {filename}")

# ----------------------------------------------------------
# 分任务 3：高级多维评估
# ----------------------------------------------------------
def task2_subtask3_advanced_evaluation(file_path):
    print("\n>>> 正在执行分任务 3：高级多维评估与灵敏度分析...")
    df = pd.read_csv(file_path)
    weekly_metrics = []
    
    grouped = df.groupby(['season', 'week'])
    for (season, week), g in grouped:
        n = len(g)
        if n < 4: continue 
        
        j_ranks = g['week_rank'].values
        j_pcts = g['judge_percentage'].values
        if 'final_est_share' in g.columns: fan_shares = g['final_est_share'].fillna(0).values
        else: fan_shares = g['voting_rate'].fillna(0).values
        fan_ranks = np.argsort(np.argsort(-fan_shares)) + 1
        
        # Simulations
        rank_sum = j_ranks + fan_ranks
        final_Rank = np.argsort(np.argsort(rank_sum)) + 1
        
        pct_sum = j_pcts + fan_shares
        final_Percent = np.argsort(np.argsort(-pct_sum)) + 1
        
        final_Rank_Save = final_Rank.copy()
        idx_last = np.where(final_Rank == n)[0][0]
        idx_2nd = np.where(final_Rank == n-1)[0][0]
        if j_ranks[idx_last] < j_ranks[idx_2nd]:
            final_Rank_Save[idx_last] = n - 1
            final_Rank_Save[idx_2nd] = n
            
        final_Percent_Save = final_Percent.copy()
        idx_last_P = np.where(final_Percent == n)[0][0]
        idx_2nd_P = np.where(final_Percent == n-1)[0][0]
        if j_pcts[idx_last_P] > j_pcts[idx_2nd_P]:
            final_Percent_Save[idx_last_P] = n - 1
            final_Percent_Save[idx_2nd_P] = n
            
        rules = ['Rank', 'Percent', 'Rank+Save', 'Percent+Save']
        finals = [final_Rank, final_Percent, final_Rank_Save, final_Percent_Save]
        
        row = {'season': season, 'week': week}
        for r, res in zip(rules, finals):
            c_fair, _ = spearmanr(res, j_ranks)
            c_fan, _ = spearmanr(res, fan_ranks)
            row[f'{r}_fairness'] = c_fair
            row[f'{r}_fan_sat'] = c_fan
            row[f'{r}_judge_risk'] = 1 if res[np.argmin(j_ranks)] == n else 0
            row[f'{r}_fan_risk'] = 1 if res[np.argmin(fan_ranks)] == n else 0
        weekly_metrics.append(row)
        
    metrics_df = pd.DataFrame(weekly_metrics)
    
    # --- 评估 1: 灵敏度分析 ---
    print(" -> 计算灵敏度分析...")
    alphas = np.linspace(0.1, 0.9, 9)
    rules = ['Rank', 'Percent', 'Rank+Save', 'Percent+Save']
    sens_res = []
    for a in alphas:
        rec = {'Fan_Weight': a}
        for r in rules:
            score = (1-a)*metrics_df[f'{r}_fairness'].mean() + a*metrics_df[f'{r}_fan_sat'].mean()
            rec[f'Score_{r}'] = score
        sens_res.append(rec)
    sens_df = pd.DataFrame(sens_res)
    print(sens_df.to_string(index=False, float_format="%.4f"))

    # --- 评估 2: 极值风险 ---
    print(" -> 计算极值风险...")
    risk_res = []
    for r in rules:
        risk_res.append({
            'Rule': r,
            'Judge Risk': metrics_df[f'{r}_judge_risk'].mean(),
            'Fan Risk': metrics_df[f'{r}_fan_risk'].mean()
        })
    risk_df = pd.DataFrame(risk_res)
    print(risk_df.to_string(index=False, float_format=lambda x: "{:.2%}".format(x)))

    return sens_df, risk_df

def task2_subtask3_visualizations(sens_df, risk_df):
    print("\n>>> 正在生成分任务 3 的可视化图表...")
    
    # 1. 灵敏度分析折线图
    plt.figure(figsize=(10, 6))
    
    # 自定义样式
    styles = {
        'Rank': {'color': '#3e7cd8', 'marker': 'o', 'ls': '-'},
        'Percent': {'color': '#ee4d7a', 'marker': 's', 'ls': '-'},
        'Rank+Save': {'color': '#3e7cd8', 'marker': '^', 'ls': '--'},
        'Percent+Save': {'color': '#ee4d7a', 'marker': 'D', 'ls': '--'}
    }
    
    for rule in ['Rank', 'Percent', 'Rank+Save', 'Percent+Save']:
        plt.plot(sens_df['Fan_Weight'], sens_df[f'Score_{rule}'], 
                 label=rule, 
                 color=styles[rule]['color'],
                 marker=styles[rule]['marker'],
                 linestyle=styles[rule]['ls'],
                 linewidth=2)
    
    plt.title('Sensitivity Analysis: Composite Score vs Fan Weight', fontsize=14)
    plt.xlabel('Fan Weight (Alpha)', fontsize=12)
    plt.ylabel('Composite Score (Fairness + Satisfaction)', fontsize=12)
    plt.legend(title='Rule System')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('Task2_Subtask3_Sensitivity.png', dpi=300)
    plt.close()
    print(" -> 已保存: Task2_Subtask3_Sensitivity.png")
    
    # 2. 极值风险分组柱状图
    plt.figure(figsize=(10, 6))
    
    # Melt 数据以适配 seaborn hue
    risk_melted = risk_df.melt(id_vars='Rule', var_name='Risk Type', value_name='Probability')
    
    # 自定义颜色: 红色代表评委风险(专业崩塌)，蓝色代表粉丝风险(人气崩塌)
    sns.barplot(x='Rule', y='Probability', hue='Risk Type', data=risk_melted, 
                palette={'Judge Risk': '#ee4d7a', 'Fan Risk': '#3e7cd8'})
    
    plt.title('Extreme Risk Analysis: Probability of Eliminating Top Contestants', fontsize=14)
    plt.ylabel('Elimination Probability', fontsize=12)
    plt.xlabel('Rule System', fontsize=12)
    plt.ylim(0, risk_melted['Probability'].max() * 1.2)
    
    # 在柱子上方标注数值
    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().annotate(f'{p.get_height():.1%}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.savefig('Task2_Subtask3_Risk.png', dpi=300)
    plt.close()
    print(" -> 已保存: Task2_Subtask3_Risk.png")

# ----------------------------------------------------------
# 主程序
# ----------------------------------------------------------
if __name__ == "__main__":
    file_path = 'Cleaned_data_with_votes.csv'
    
    # 1. 运行分任务 1 (含可视化)
    task2_subtask1_rules_comparison(file_path)
    
    # 2. 运行分任务 2 (含分析与可视化)
    targets = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]
    df_cases = task2_subtask2_controversial_analysis(file_path, targets)
    df_cases.to_csv("Task2_Controversial_Analysis.csv", index=False)
    
    # 修改点：读取 CSV 后会自动按 (Celebrity, Season) 分开画图
    task2_subtask2_visualize_timelines("Task2_Controversial_Analysis.csv")
    
# 3. 运行分任务 3 (含分析与可视化)
    sens_df, risk_df = task2_subtask3_advanced_evaluation(file_path)
    task2_subtask3_visualizations(sens_df, risk_df)