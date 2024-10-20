import pandas as pd

# 读取CSV文件
df = pd.read_csv('data1.csv')

# 提取各列数据并命名变量
column_1 = df['x'].values
column_2 = df['rho'].values
column_3 = df['P'].values
column_4 = df['u'].values
column_5 = df['e'].values
column_6 = df['M'].values
column_7 = df['H'].values
