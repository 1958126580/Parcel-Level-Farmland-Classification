import pandas as pd
import numpy as np

# i = [0,1,3,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,57,58,59,60,62,63,64,67,68]
i = [14,50,53,59]

for file_index in i:
    # 读取CSV文件
    file_path = fr"D:\0-code\AGRS_time_series-main\data\Hetian_TEST\test_{file_index}.csv"  # 替换为你的文件路径
    df = pd.read_csv(file_path)

    # 定义doy列表
    doy_list = [3, 18, 23, 43, 58, 68, 83, 93, 98, 113, 123, 128, 138, 163, 183, 188, 198, 208, 238, 248, 268, 278, 283,
                288, 303, 308, 313, 328, 333, 338]

    # 拆分row_col为gridcode和row_col
    df[['gridcode', 'row_col']] = df['row_col'].str.split('_', n=1, expand=True)
    df['gridcode'] = file_index

    # 获取所有现有的doy
    existing_doy = sorted([int(col.split('_')[1]) for col in df.columns if col.startswith('NDVI_')])


    # 插值函数
    def interpolate_value(row, feature, target_doy):
        # 找到目标doy的前后doy
        before_doy = max([d for d in existing_doy if d < target_doy], default=None)
        after_doy = min([d for d in existing_doy if d > target_doy], default=None)

        if before_doy is None or after_doy is None:
            raise ValueError(f"Cannot interpolate for doy {target_doy}, no valid surrounding doy values.")

        before_col = f"{feature}_{before_doy}"
        after_col = f"{feature}_{after_doy}"

        # 插值公式
        interpolated_value = row[before_col] + (row[after_col] - row[before_col]) * (target_doy - before_doy) / (
                    after_doy - before_doy)
        # 保留四位小数
        row = round(interpolated_value, 4)
        return row


    # 创建新的DataFrame
    new_columns = ['gridcode', 'row_col', 'cropcode', 'objId']
    ndvi_columns = [f"NDVI_{doy}" for doy in doy_list]
    ndre1_columns = [f"NDRE1_{doy}" for doy in doy_list]

    new_df = df[['gridcode', 'row_col', 'cropcode', 'objId']].copy()

    # 填充NDVI和NDRE1的值
    for doy in doy_list:
        ndvi_col = f"NDVI_{doy}"
        ndre1_col = f"NDRE1_{doy}"

        if ndvi_col in df.columns:
            new_df[ndvi_col] = df[ndvi_col]
        else:
            new_df[ndvi_col] = df.apply(lambda row: interpolate_value(row, 'NDVI', doy), axis=1)

        if ndre1_col in df.columns:
            new_df[ndre1_col] = df[ndre1_col]
        else:
            new_df[ndre1_col] = df.apply(lambda row: interpolate_value(row, 'NDRE1', doy), axis=1)

    # 重新排列列顺序
    new_df = new_df[['gridcode', 'row_col', 'cropcode', 'objId'] + ndvi_columns + ndre1_columns]

    # 保存结果到新的CSV文件
    new_df.to_csv(fr"D:\0-code\AGRS_time_series-main\data\Hetian_TEST\test_{file_index}_modified.csv", index=False)
    print(f"已完成 i={file_index}")
