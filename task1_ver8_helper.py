import pandas as pd
df = pd.read_csv('Cleaned_data_with_votes.csv',
                 encoding='utf-8')  # 若报错可改为 'gbk'
mean_cv = df['vote_cv'].mean()  # 会自动跳过 NaN
print(mean_cv)
