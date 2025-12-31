### 将anaconda中得到的逐像素时序特征转为地块的时序特征，同时整理为转为tsv格式的csv ###

import pandas as pd

# 读取CSV文件（anaconda处理得到的结果）
df = pd.read_csv(r"D:\0-code\AGRS_time_series-main\zyh\luorun_train.csv")
df2 = pd.read_csv(r"D:\0-code\AGRS_time_series-main\zyh\luorun_val.csv")

# 1. 删除“Row_Column”列
df.drop('Row_Column', axis=1, inplace=True)
df2.drop('Row_Column', axis=1, inplace=True)

# 2. 对于每个相同的“number”，计算每个时间（即每一列）的特征值的平均值
grouped = df.groupby('number')
grouped2 = df2.groupby('number')
averaged_features = grouped.mean().reset_index()
averaged_features2 = grouped2.mean().reset_index()

# 3. 将表格整理格式
# 将label列移到最后
averaged_features = averaged_features.reindex(columns=['number'] + [col for col in averaged_features.columns if col != 'number' and col != 'label'] + ['label'])
averaged_features2 = averaged_features2.reindex(columns=['number'] + [col for col in averaged_features2.columns if col != 'number' and col != 'label'] + ['label'])

# 输出带有地块编码的CSV文件，用于将时序数据填入地块属性表
averaged_features.to_csv('luorun_train_bynumber.csv', index=False)
averaged_features2.to_csv('luorun_val_bynumber.csv', index=False)

# 删除“number”列
averaged_features.drop('number', axis=1, inplace=True)
averaged_features2.drop('number', axis=1, inplace=True)

# 输出CSV文件，用于转换为ts格式（与本框架原始数据格式保持一致）进行训练预测
averaged_features.to_csv('luorun_train_withoutnumber.csv', index=False, header=True)
averaged_features2.to_csv('luorun_val_withoutnumber.csv', index=False, header=True)

# 打印结果以验证
# print(averaged_features.head())