import csv
from collections import defaultdict

def count_ndvi_values(csv_file_path):

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # 读取表头

        # 提取所有doy值
        doy_columns = [col for col in header if col.startswith('NDVI_')]
        doy_values = [int(col.split('_')[1]) for col in doy_columns]

        # 初始化一个字典来统计每个doy下NDVI有值的数量
        ndvi_count = defaultdict(int)

        # 遍历每一行数据
        for row in reader:
            for col, doy in zip(doy_columns, doy_values):
                ndvi_value = row[header.index(col)]
                if ndvi_value:  # 如果NDVI值不为空
                    ndvi_count[doy] += 1

    # 根据NDVI有值的数量从多到少排序
    sorted_doy = sorted(ndvi_count.items(), key=lambda x: x[1], reverse=True)

    # 输出排序后的列表
    print("排序后的doy列表（按有特征值的数量从多到少）：")
    for doy, count in sorted_doy:
        print(f"doy: {doy}, 有特征值的数量: {count}")

    # 输出只包含doy值的从多到少的列表
    sorted_doy_list = [doy for doy, count in sorted_doy]
    print("从多到少的doy列表：")
    print(sorted_doy_list)

# 调用函数
csv_file_path = r"D:\0-code\AGRS_time_series-main\data\Hetian\val.csv" # 替换为你的CSV文件路径
count_ndvi_values(csv_file_path)