'''
用sg滤波对插值后的时序曲线进行平滑

window_length：窗口长度，必须是奇数。它决定了在每个数据点周围考虑多少个数据点。较大的窗口长度会导致更平滑的结果，但可能会丢失更多的细节。
polyorder：多项式的阶数。它决定了拟合多项式的复杂程度。较高的阶数可以更好地拟合数据，但可能会导致过拟合。
'''
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
import time  # 导入time模块

start_time = time.time()  # 记录每个i开始的时间

# i = [0,1,3,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,57,58,59,60,62,63,64,67,68]
i = [8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,57,58,59,60,62,63,64,67,68]

for file_index in i:
    # 输入文件路径
    file_path = fr"D:\0-code\AGRS_time_series-main\data\Hetian_TEST\test_{file_index}_modified.csv"
    # 输出文件路径
    output_file_path = fr"D:\0-code\AGRS_time_series-main\data\Hetian_TEST\test_{file_index}_modified_smoothed.csv"

    # 分块大小（可以根据内存大小调整）
    chunk_size = 10000

    # 读取第一块数据以获取列名
    first_chunk = next(pd.read_csv(file_path, chunksize=chunk_size))
    ndvi_columns = [col for col in first_chunk.columns if col.startswith('NDVI')]
    ndre1_columns = [col for col in first_chunk.columns if col.startswith('NDRE1')]
    ndvi_smoothed_columns = [col.replace('NDVI', 'NDVI_SG') for col in ndvi_columns]
    ndre1_smoothed_columns = [col.replace('NDRE1', 'NDRE1_SG') for col in ndre1_columns]

    # 确保输出文件不存在（避免覆盖）
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # 分块读取和处理数据
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # 初始化新的平滑列
        for col in ndvi_smoothed_columns + ndre1_smoothed_columns:
            chunk[col] = np.nan

        # 对每一行进行 Savitzky-Golay 滤波
        for idx, row in chunk.iterrows():
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
            chunk.loc[idx, ndvi_smoothed_columns] = np.round(ndvi_smoothed, 4)
            chunk.loc[idx, ndre1_smoothed_columns] = np.round(ndre1_smoothed, 4)

        # 删除原始的 NDVI 和 NDRE1 列
        chunk.drop(columns=ndvi_columns + ndre1_columns, inplace=True)

        # 将处理后的数据保存到输出文件
        # 如果是第一块数据，写入表头；否则追加数据
        chunk.to_csv(output_file_path, mode='a', header=not os.path.exists(output_file_path), index=False)

    end_time = time.time()  # 记录每个i结束的时间
    elapsed_time = (end_time - start_time) / 60  # 将秒转换为分钟
    print(f"已完成i={file_index}，所用时长：{elapsed_time:.2f}分钟")