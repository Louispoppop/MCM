import pandas as pd
import numpy as np
import re

def clean_dwts_data_final(file_path):
    """
    清洗 2026 MCM Problem C 的 DWTS 数据集。
    版本: V2 (包含 placement, 国籍, 居住地, 且修正了 Percentage 计算逻辑)
    """
    print(f"正在读取数据: {file_path} ...")
    df = pd.read_csv(file_path)

    # ==========================================
    # 1. 转换数据格式 (Wide to Long)
    # ==========================================
    # 定义不需要转换的“元数据”列，显式包含 placement
    id_vars = [
        'celebrity_name', 
        'ballroom_partner', 
        'celebrity_industry', 
        'celebrity_homestate',          
        'celebrity_homecountry/region', 
        'celebrity_age_during_season', 
        'season', 
        'results', 
        'placement'  # <--- 关键修改：保留最终排名
    ]
    
    # 找出所有包含 'judge' 的评分列
    score_cols = [c for c in df.columns if 'judge' in c]
    
    # Melt 操作：宽表变长表
    df_long = df.melt(id_vars=id_vars, value_vars=score_cols, 
                      var_name='week_judge_str', value_name='score')

    # ==========================================
    # 2. 提取周数和评委编号
    # ==========================================
    pattern = r'week(\d+)_judge(\d+)_score'
    extracted = df_long['week_judge_str'].str.extract(pattern)
    df_long['week'] = pd.to_numeric(extracted[0])
    df_long['judge_num'] = pd.to_numeric(extracted[1])
    
    # 将分数转换为数值型
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')

    # ==========================================
    # 3. 数据过滤与汇总
    # ==========================================
    # 去除无效评分行
    df_clean = df_long.dropna(subset=['score'])
    
    # 按周汇总分数
    weekly_stats = df_clean.groupby(['season', 'celebrity_name', 'week']).agg(
        total_judge_score=('score', 'sum'),
        num_judges=('score', 'count'),
        results=('results', 'first')
    ).reset_index()

    # 过滤掉总分为0的无效行
    weekly_stats = weekly_stats[weekly_stats['total_judge_score'] > 0].copy()

    # ==========================================
    # 4. 智能解析真实淘汰周
    # ==========================================
    def get_elimination_week(res_str):
        if pd.isna(res_str):
            return 999
        res_str = str(res_str)
        # 匹配标准淘汰格式 "Eliminated Week X"
        match = re.search(r'Eliminated Week (\d+)', res_str)
        if match:
            return int(match.group(1))
        # 处理退赛
        if "Withdrew" in res_str:
            return -1 
        # 决赛选手
        if "Place" in res_str:
            return 999 
        return 999

    weekly_stats['parsed_elim_week'] = weekly_stats['results'].apply(get_elimination_week)

    # 修正退赛逻辑
    max_weeks = weekly_stats.groupby(['season', 'celebrity_name'])['week'].max().reset_index()
    max_weeks.rename(columns={'week': 'last_active_week'}, inplace=True)
    weekly_stats = weekly_stats.merge(max_weeks, on=['season', 'celebrity_name'], how='left')
    
    weekly_stats['true_elimination_week'] = weekly_stats.apply(
        lambda row: row['last_active_week'] if row['parsed_elim_week'] == -1 else row['parsed_elim_week'], 
        axis=1
    )

    # ==========================================
    # 5. 构建目标变量
    # ==========================================
    weekly_stats['is_eliminated'] = (weekly_stats['week'] == weekly_stats['true_elimination_week']).astype(int)

    # ==========================================
    # 6. 计算关键特征
    # ==========================================
    
    # A. 计算当周总分池 (用于 Percentage Method)
    week_total_scores = weekly_stats.groupby(['season', 'week'])['total_judge_score'].transform('sum')
    
    # B. 计算得分份额
    weekly_stats['judge_percentage'] = weekly_stats['total_judge_score'] / week_total_scores
    
    # C. 计算绝对得分率 (可选)
    weekly_stats['score_rate'] = weekly_stats['total_judge_score'] / (weekly_stats['num_judges'] * 10)
    
    # D. 计算当周排名
    weekly_stats['week_rank'] = weekly_stats.groupby(['season', 'week'])['total_judge_score'] \
                                            .rank(ascending=False, method='min')

    # ==========================================
    # 7. 合并所有元数据 (关键：包含 placement)
    # ==========================================
    meta_cols_list = [
        'season', 
        'celebrity_name', 
        'celebrity_industry', 
        'celebrity_age_during_season',
        'celebrity_homestate',          
        'celebrity_homecountry/region', 
        'ballroom_partner',
        'placement'   # <--- 关键修改：合并回 placement
    ]
    
    meta_cols = df[meta_cols_list].drop_duplicates()
    final_df = weekly_stats.merge(meta_cols, on=['season', 'celebrity_name'], how='left')

    # 将 placement 转为数值 (1, 2, 3...)，非数字(如退赛)转为 NaN
    final_df['placement'] = pd.to_numeric(final_df['placement'], errors='coerce')

    # 排序
    final_df = final_df.sort_values(['season', 'week', 'week_rank'])

    print("数据清洗完成！已包含 'placement' 列。")
    return final_df

# --- 执行部分 ---
if __name__ == "__main__":
    input_file = '2026_MCM_Problem_C_Data.csv' 
    output_file = 'Cleaned_data_with_placement.csv'
    
    try:
        cleaned_data = clean_dwts_data_final(input_file)
        
        # 预览结果，重点检查 placement
        print("\n数据预览 (决赛周选手):")
        # 筛选出第1赛季最后一周的数据查看
        max_week_s1 = cleaned_data[cleaned_data['season']==1]['week'].max()
        preview = cleaned_data[(cleaned_data['season']==1) & (cleaned_data['week']==max_week_s1)]
        print(preview[['season', 'week', 'celebrity_name', 'results', 'placement']])
        
        # 保存
        cleaned_data.to_csv(output_file, index=False)
        print(f"\n文件已保存至: {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}。")
    except Exception as e:
        print(f"发生错误: {e}")