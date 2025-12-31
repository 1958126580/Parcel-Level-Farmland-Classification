### 将csv转为ts格式文件 ###
import pandas as pd

# 读取CSV文件
df = pd.read_csv('deeplearning_test_withoutnumber_ndvi.csv')

# 准备一个新的列表来存储转换后的数据行
converted_data = []

# 遍历数据帧中的每一行
for index, row in df.iterrows():
    # 将前19个特征值转换为逗号分隔的字符串，并添加到列表中
    # 替换19为时间序列长度*特征数
    features_str = ','.join(map(str, row[:19].tolist()))
    # 添加类别标签到行的末尾
    label = row[19]
    # 将特征值和标签组合成一行字符串
    data_line = f"{features_str}:{label}\n"
    # 添加到转换后的数据列表中
    converted_data.append(data_line)

# 将转换后的数据写入新的文本文件
with open('WeiganFarmland_TEST.ts', 'w') as f:
    f.writelines(converted_data)