### 将表格特征值填入shp ###

import geopandas as gpd
import pandas as pd

# 读取 cropland.shp 文件
shp_data = gpd.read_file(r'E:\huajiang_crop\huajiang_sfq_reprj.shp')
# 读取 csv 文件
csv_data = pd.read_csv('predict_result.csv')

# 遍历 csv 数据的列名
for column in csv_data.columns:
    # 如果列名不是 number 或 label
    if column not in ['number', 'label']:
        # 在 shp 文件中添加新的字段，字段类型为 float64
        shp_data[column] = None

# 遍历 csv 数据的每一行
for _, row in csv_data.iterrows():
    # 获取 number 列的值
    number_value = row['number']

    # 查找 shp 文件中 number 字段等于 number_value 的行索引
    index = shp_data[shp_data['number'] == number_value].index

    # 如果找到了匹配的行
    if not index.empty:
        # 获取匹配行的索引
        index = index[0]
        # 遍历 csv 数据的列名
        for column in csv_data.columns:
            # 如果列名不是 number 或 label
            if column not in ['number', 'label']:
                # 将 csv 数据中对应的值赋给 shp 文件中对应的字段
                shp_data.at[index, column] = row[column]
            else:
                # 将 csv 数据中对应的值赋给 shp 文件中对应的字段
                shp_data.at[index, column] = row[column]

# 将修改后的数据保存为新的 shapefile 文件
shp_data.to_file(r'E:\huajiang_crop\huajiang_sfq_classification.shp')