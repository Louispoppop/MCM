import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# 读取数据
file_path = r'd:\美赛\Cleaned_data_with_votes.csv'
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
    exit()

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"读取文件失败: {e}")
    exit()

# 筛选 Season 2
target_season = 2
target_names = ['Jerry Rice', 'Drew Lachey']

# 确保只选出这两个人
df_filtered = df[(df['season'] == target_season) & (
    df['celebrity_name'].isin(target_names))].copy()

if df_filtered.empty:
    print("未找到 Season 2 中 Jerry Rice 和 Drew Lachey 的数据。")
    exit()

# 排序
df_filtered.sort_values(by='week', inplace=True)

# 准备绘图
# 使用两个子图，共享x轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 颜色和样式
colors = {
    'Jerry Rice': "#ee4d7a",
    'Drew Lachey': "#3e7cd8"
}

# 准备条形图参数
weeks = sorted(df_filtered['week'].unique())
n_weeks = len(weeks)
bar_width = 0.35
indices = np.arange(len(weeks))

# 获取每个人的数据，保证对齐
jerry_data = df_filtered[df_filtered['celebrity_name']
                         == 'Jerry Rice'].set_index('week').reindex(weeks)
drew_data = df_filtered[df_filtered['celebrity_name']
                        == 'Drew Lachey'].set_index('week').reindex(weeks)

# ---------------------------------------------------------
# Subplot 1: Judge Scores (Bar Chart)
# ---------------------------------------------------------

# 绘制柱状图
# Handle NaN by fillna(0) for plotting purposes, but we will check heights for text
heights1 = jerry_data['total_judge_score'].fillna(0)
heights2 = drew_data['total_judge_score'].fillna(0)

bars1_jerry = ax1.bar(indices - bar_width/2, heights1, bar_width,
                      label='Jerry Rice', color=colors['Jerry Rice'], alpha=0.9, edgecolor='black', linewidth=0.5)
bars1_drew = ax1.bar(indices + bar_width/2, heights2, bar_width,
                     label='Drew Lachey', color=colors['Drew Lachey'], alpha=0.9, edgecolor='black', linewidth=0.5)

# 标出数值


def add_labels(ax, bars, original_data, fmt="{:.0f}"):
    """
    ax: axes object
    bars: bar container
    original_data: series with original data (to check for NaN)
    fmt: format string
    """
    # Convert numpy array or list to list for enumeration if needed, but bars is iterable
    for i, rect in enumerate(bars):
        height = rect.get_height()
        # Original value check
        val = original_data.iloc[i]
        if pd.isna(val) or val == 0:
            # Don't label if data is missing or 0 (unless 0 is valid, but here we assume scores/votes > 0)
            continue

        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')


add_labels(ax1, bars1_jerry, jerry_data['total_judge_score'], fmt="{:.0f}")
add_labels(ax1, bars1_drew, drew_data['total_judge_score'], fmt="{:.0f}")

ax1.set_ylabel('Judge Score', fontsize=12)  # English label
ax1.set_title('Judge Scores',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim(0, 35)  # Adjust based on max score 30

# ---------------------------------------------------------
# Subplot 2: Vote Shares (Bar Chart)
# ---------------------------------------------------------

# 确定使用哪一列
if 'final_est_share' in df_filtered.columns:
    col_name = 'final_est_share'
    y_label_text = 'Estimated Vote Share'
else:
    col_name = 'judge_percentage'
    y_label_text = 'Score Percentage'

heights3 = jerry_data[col_name].fillna(0)
heights4 = drew_data[col_name].fillna(0)

# 绘制柱状图
bars2_jerry = ax2.bar(indices - bar_width/2, heights3, bar_width,
                      label='Jerry Rice', color=colors['Jerry Rice'], alpha=0.9, edgecolor='black', linewidth=0.5)
bars2_drew = ax2.bar(indices + bar_width/2, heights4, bar_width,
                     label='Drew Lachey', color=colors['Drew Lachey'], alpha=0.9, edgecolor='black', linewidth=0.5)

# 标出百分比数值 (保留两位小数)
add_labels(ax2, bars2_jerry, jerry_data[col_name], fmt="{:.2%}")
add_labels(ax2, bars2_drew, drew_data[col_name], fmt="{:.2%}")

ax2.set_ylabel(y_label_text, fontsize=12)
ax2.set_xlabel('Week', fontsize=12)
ax2.set_title('Estimated Vote Share',
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
ax2.set_ylim(0, 0.5)  # Adjust based on max share ~35-40%

# 设置X轴刻度
ax2.set_xticks(indices)
ax2.set_xticklabels(weeks, fontsize=11)

fig.suptitle('Season 2: Jerry Rice (2nd) vs Drew Lachey (1st)',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存
output_path = r'd:\美赛\season2_jerry vs drew.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Success! Bar chart saved to: {output_path}")

# Plot2

file_path = r'd:\美赛\season2_final_result.csv'
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
    exit()

df = pd.read_csv(file_path)

# 确认列存在
if 'judge_percentage' not in df.columns or 'vote_share_est' not in df.columns:
    print("文件中缺少 judge_percentage 或 vote_share_est 列。")
    exit()

names = df['celebrity_name'].tolist()
judge_vals = df['judge_percentage'].astype(float).tolist()
vote_vals = df['vote_share_est'].astype(float).tolist()

# 为每个饼图突出最大值


def explode_for_max(vals):
    max_idx = int(pd.Series(vals).idxmax())
    explode = [0.05 if i == max_idx else 0.0 for i in range(len(vals))]
    return explode


explode_j = explode_for_max(judge_vals)
explode_v = explode_for_max(vote_vals)

colors = ["#3e7cd8", "#9C86DA", "#ee4d7a"]  # 可按需调整

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Season 2 Final: Judge & Vote Percentage',
             fontsize=16, fontweight='bold')

# 评委评分占比
axes[0].pie(judge_vals, labels=names, autopct='%1.1f%%', startangle=90,
            explode=explode_j, colors=colors[:len(names)], wedgeprops=dict(edgecolor='w'))
axes[0].set_title('Judge Percentage')

# Vote Share
axes[1].pie(vote_vals, labels=names, autopct='%1.1f%%', startangle=90,
            explode=explode_v, colors=colors[:len(names)], wedgeprops=dict(edgecolor='w'))
axes[1].set_title('Vote Share (Estimated)')

plt.tight_layout(rect=[0, 0, 1, 0.94])

output_path = r'd:\美赛\season2_final_pie.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"已保存：{output_path}")
