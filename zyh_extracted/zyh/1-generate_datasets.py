import csv
import pandas as pd
import numpy as np

# 按照地块分训练集测试集
sampledata = r"C:\zyh\ktywork\20250703sichuan_agriculture\sichuan\time_series\luorun\cty\0\pixel_sample.txt"

# Function to calculate NDVI
def calculate_ndvi(b1, b4):
    return (b4 - b1) / (b4 + b1)

# Function to calculate NDRE1
def calculate_ndre1(b2, b3):
    return (b3 - b2) / (b3 + b2)

# Read the txt file and process the data
with open(sampledata, 'r') as file:
    data = file.readlines()

output_data = []

# ndvindvi……（n个）ndrendre……（n个）
for line in data:
    line_data = line.strip().split(',')
    row_col = line_data[0].split('_')
    crop_type = line_data[1]
    num = line_data[2]
    ndvi_values = []
    ndre1_values = []
    for i in range(3, len(line_data)-4, 5):
        date = line_data[i]
        b1, b2, b3, b4 = int(line_data[i+1]), int(line_data[i+2]), int(line_data[i+3]), int(line_data[i+4])
        ndvi = calculate_ndvi(b1, b4)
        ndre1 = calculate_ndre1(b2, b3)
        ndvi_values.append(ndvi)
        ndre1_values.append(ndre1)
    output_data.append([f"{row_col[1]}_{row_col[2]}",crop_type,num] + ndvi_values + ndre1_values)
# Write the output to a csv file
with open('luorun_sample.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Row_Column', 'label', 'number'] + [f'NDVI_{i}' for i in range(1, 8)] + [f'NDRE1_{i}' for i in range(1, 8)])
    writer.writerows(output_data)

## ndvi ndre ndvi ndre……
# for line in data:
#     line_data = line.strip().split(',')
#     row_col = line_data[0].split('_')
#     crop_type = line_data[1]
#     num = line_data[2]
#     ndvi_values = []
#     ndre1_values = []
#     for i in range(3, len(line_data) - 4, 5):
#         date = line_data[i]
#         b1, b2, b3, b4 = int(line_data[i + 1]), int(line_data[i + 2]), int(line_data[i + 3]), int(line_data[i + 4])
#         ndvi = calculate_ndvi(b1, b4)
#         ndre1 = calculate_ndre1(b2, b3)
#
#         # Append NDVI and NDRE1 values to the respective lists
#         ndvi_values.append(ndvi)
#         ndre1_values.append(ndre1)
#
#         # Interleave NDVI and NDRE1 values for the output
#     interleaved_values = []
#     for ndvi, ndre1 in zip(ndvi_values, ndre1_values):
#         interleaved_values.append(ndvi)
#         interleaved_values.append(ndre1)
#         # Append the row data with the interleaved NDVI and NDRE1 values
#     output_data.append([f"{row_col[1]}_{row_col[2]}",crop_type,num] + interleaved_values)
# # Write the output to a csv file
# with open('deeplearning_train.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # Write the header row with the correct column names
#     header = ['Row_Column', 'label', 'number']
#     for i in range(1, 20):
#         header.append(f'NDVI_{i}')
#         header.append(f'NDRE1_{i}')
#     writer.writerow(header)
#     # Write the data rows
#     writer.writerows(output_data)



# 按照num按比例2：8划分数据集
df = pd.read_csv('luorun_sample.csv')

# 根据label分组
grouped = df.groupby('label')

# 创建训练和测试数据集的索引
train_indices = []
test_indices = []

# 遍历每个Typecode分组
for label, group in grouped:
    # 获取该Typecode下所有num的列表
    nums = group['number'].unique()
    # 计算num的总数
    num_count = len(nums)
    # 选择20%的num作为测试集
    test_num_count = int(num_count * 0.2)
    # 随机选择test_num_count个num
    test_nums = np.random.choice(nums, size=test_num_count, replace=False)

    # 保留测试集num的索引
    test_indices.extend(group[group['number'].isin(test_nums)].index.tolist())

    # 剩余像素作为训练集
    train_indices.extend(group[~group['number'].isin(test_nums)].index.tolist())

# 创建训练数据集
train_dataset = df.loc[train_indices].sort_index()

# 创建测试数据集
test_dataset = df.loc[test_indices].sort_index()

# 将训练和测试数据集写入csv文件
train_dataset.to_csv('luorun_train.csv', index=False)
test_dataset.to_csv('luorun_val.csv', index=False)


# # #按照像元划分数据集
# # Read the output csv file
# df = pd.read_csv('sample_6sorts_new.csv')
#
# # Split the data based on Crop_Type
# grouped = df.groupby('Typecode', sort=False)
#
# # Create train and test datasets
# train_indices = []
# test_indices = []
#
# for name, group in grouped:
#     n = group.shape[0]
#     test_size = int(n * 0.2)
#     all_indices = group.index.tolist()
#     test_indices.extend(np.random.choice(all_indices, size=test_size, replace=False))
#     train_indices.extend([idx for idx in all_indices if idx not in test_indices])
#
# # Create train dataset without sorting
# train_dataset = df.loc[train_indices].sort_index()
#
# # Create test dataset without sorting
# test_dataset = df.loc[test_indices].sort_index()
#
# # Write the train and test datasets to csv files
# train_dataset.to_csv('train_datasat_6sorts_newxy.csv', index=False)
# test_dataset.to_csv('test_datasat_6sorts_newxy.csv', index=False)



### 格式：19个ndvi+19个ndre1
# sampledata = r"G:\yb_txt\0\sample_merged.txt"
#
# # Function to calculate NDVI
# def calculate_ndvi(b1, b4):
#     return (b4 - b1) / (b4 + b1)
#
# # Function to calculate NDRE1
# def calculate_ndre1(b2, b3):
#     return (b3 - b2) / (b3 + b2)
#
# # Read the txt file and process the data
# with open(sampledata, 'r') as file:
#     data = file.readlines()
#
# output_data = []
#
# for line in data:
#     line_data = line.strip().split(',')
#     row_col = line_data[0].split('_')
#     crop_type = line_data[1]
#     ndvi_values = []
#     ndre1_values = []
#     for i in range(3, len(line_data)-4, 5):
#         date = line_data[i]
#         b1, b2, b3, b4 = int(line_data[i+1]), int(line_data[i+2]), int(line_data[i+3]), int(line_data[i+4])
#         ndvi = calculate_ndvi(b1, b4)
#         ndre1 = calculate_ndre1(b2, b3)
#         ndvi_values.append(ndvi)
#         ndre1_values.append(ndre1)
#     output_data.append([f"{row_col[1]}_{row_col[2]}", crop_type] + ndvi_values + ndre1_values)
#
# # Write the output to a csv file
# with open('sample.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Row_Column', 'Typecode'] + [f'NDVI_{i}' for i in range(1, 20)] + [f'NDRE1_{i}' for i in range(1, 20)])
#     writer.writerows(output_data)

# # Read the output csv file
# df = pd.read_csv('sample.csv')
#
# # Split the data based on Crop_Type
# grouped = df.groupby('Typecode', sort=False)
#
# # Create train and test datasets
# train_indices = []
# test_indices = []
#
# for name, group in grouped:
#     n = group.shape[0]
#     test_size = int(n * 0.2)
#     all_indices = group.index.tolist()
#     test_indices.extend(np.random.choice(all_indices, size=test_size, replace=False))
#     train_indices.extend([idx for idx in all_indices if idx not in test_indices])
#
# # Create train dataset without sorting
# train_dataset = df.loc[train_indices].sort_index()
#
# # Create test dataset without sorting
# test_dataset = df.loc[test_indices].sort_index()
#
# # Write the train and test datasets to csv files
# train_dataset.to_csv('train_datasat0206.csv', index=False)
# test_dataset.to_csv('test_datasat0206.csv', index=False)