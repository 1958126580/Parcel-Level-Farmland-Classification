import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

'''
用sg滤波对插值后的时序曲线进行平滑

window_length：窗口长度，必须是奇数。它决定了在每个数据点周围考虑多少个数据点。较大的窗口长度会导致更平滑的结果，但可能会丢失更多的细节。
polyorder：多项式的阶数。它决定了拟合多项式的复杂程度。较高的阶数可以更好地拟合数据，但可能会导致过拟合。
'''

# 读取CSV文件
file_path = r"D:\0-code\AGRS_time_series-main\data\Hetian_TRAIN\experiment_3（8_2_0）\train_modified_interpolated.csv"
data = pd.read_csv(file_path)

# 提取NDVI和NDRE1的列名
ndvi_columns = [col for col in data.columns if col.startswith('NDVI')]
ndre1_columns = [col for col in data.columns if col.startswith('NDRE1')]

# 创建新的列名
ndvi_smoothed_columns = [col.replace('NDVI', 'NDVI_SG') for col in ndvi_columns]
ndre1_smoothed_columns = [col.replace('NDRE1', 'NDRE1_SG') for col in ndre1_columns]

# 初始化新的列
for col in ndvi_smoothed_columns + ndre1_smoothed_columns:
    data[col] = np.nan

# 对每个像素进行Savitzky-Golay滤波
for idx, row in data.iterrows():
    ndvi_values = row[ndvi_columns].values
    ndre1_values = row[ndre1_columns].values

    # 确保数据长度足够进行滤波
    if len(ndvi_values) > 10:
        ndvi_smoothed = savgol_filter(ndvi_values, window_length=5, polyorder=2)
        ndre1_smoothed = savgol_filter(ndre1_values, window_length=5, polyorder=2)
    else:
        ndvi_smoothed = ndvi_values
        ndre1_smoothed = ndre1_values

    # 保留四位小数
    data.loc[idx, ndvi_smoothed_columns] = np.round(ndvi_smoothed, 4)
    data.loc[idx, ndre1_smoothed_columns] = np.round(ndre1_smoothed, 4)

# 删除原始的NDVI和NDRE1列
data.drop(columns=ndvi_columns + ndre1_columns, inplace=True)

# 保存到新的CSV文件
output_file_path = r"D:\0-code\AGRS_time_series-main\data\Hetian_TRAIN\experiment_3（8_2_0）\train_modified_interpolated_smoothed.csv"
data.to_csv(output_file_path, index=False)

# '''
# 出图 按作物类型
# 对比平滑前后曲线情况
# '''
#
# # 绘制平滑前后的曲线图
# def plot_smoothed_curves(original_data, smoothed_data, crop_code):
#     # 筛选出特定cropcode的数据
#     original_crop_data = original_data[original_data['cropcode'] == crop_code]
#     smoothed_crop_data = smoothed_data[smoothed_data['cropcode'] == crop_code]
#
#     # 计算每个DOY的平均值
#     ndvi_means = original_crop_data[ndvi_columns].mean().values
#     ndvi_smoothed_means = smoothed_crop_data[ndvi_smoothed_columns].mean().values
#     ndre1_means = original_crop_data[ndre1_columns].mean().values
#     ndre1_smoothed_means = smoothed_crop_data[ndre1_smoothed_columns].mean().values
#
#     # 提取DOY
#     doy = [int(col.split('_')[-1]) for col in ndvi_columns]
#
#     # 绘制NDVI曲线
#     plt.figure(figsize=(12, 6))
#     plt.plot(doy, ndvi_means, label='NDVI (Original)', linestyle='--')
#     plt.plot(doy, ndvi_smoothed_means, label='NDVI (Smoothed)', linestyle='-')
#     plt.title(f'NDVI for Crop Code {crop_code}')
#     plt.xlabel('Day of Year (DOY)')
#     plt.ylabel('NDVI')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#     # 绘制NDRE1曲线
#     plt.figure(figsize=(12, 6))
#     plt.plot(doy, ndre1_means, label='NDRE1 (Original)', linestyle='--')
#     plt.plot(doy, ndre1_smoothed_means, label='NDRE1 (Smoothed)', linestyle='-')
#     plt.title(f'NDRE1 for Crop Code {crop_code}')
#     plt.xlabel('Day of Year (DOY)')
#     plt.ylabel('NDRE1')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# # 重新读取原始数据用于绘图
# original_data = pd.read_csv(file_path)
#
# # 绘制各个cropcode的曲线
# plot_smoothed_curves(original_data, data, crop_code=1)
# plot_smoothed_curves(original_data, data, crop_code=2)
# plot_smoothed_curves(original_data, data, crop_code=3)
# plot_smoothed_curves(original_data, data, crop_code=4)
# plot_smoothed_curves(original_data, data, crop_code=5)
# plot_smoothed_curves(original_data, data, crop_code=6)
# plot_smoothed_curves(original_data, data, crop_code=7)
# plot_smoothed_curves(original_data, data, crop_code=8)

