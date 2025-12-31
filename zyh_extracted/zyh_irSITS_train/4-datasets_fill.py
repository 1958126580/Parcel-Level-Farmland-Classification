import pandas as pd

file_path = r"D:\0-code\AGRS_time_series-main\data\Hetian_TRAIN\experiment_3（8_2_0）\val_modified.csv"

# 读取CSV文件
df = pd.read_csv(file_path)

# 定义插值函数
def interpolate_values(row, prefix):
    # 获取所有以指定前缀开头的列
    columns = [col for col in df.columns if col.startswith(prefix)]
    columns.sort(key=lambda x: int(x.split('_')[1]))  # 按DOY排序

    for i, col in enumerate(columns):
        if pd.isna(row[col]):
            # 找到当前缺失值的前后最近的非空值
            prev_col = next((c for c in columns[:i][::-1] if not pd.isna(row[c])), None)
            next_col = next((c for c in columns[i + 1:] if not pd.isna(row[c])), None)

            if prev_col and next_col:
                # 计算DOY差值
                prev_doy = int(prev_col.split('_')[1])
                next_doy = int(next_col.split('_')[1])
                current_doy = int(col.split('_')[1])

                # 插值公式
                interpolated_value = (row[prev_col] * (next_doy - current_doy) + row[next_col] * (current_doy - prev_doy)) / (next_doy - prev_doy)
                # 保留四位小数
                row[col] = round(interpolated_value, 4)

    return row

# 对NDVI和NDRE1列进行插值
df = df.apply(lambda row: interpolate_values(row, 'NDVI'), axis=1)
df = df.apply(lambda row: interpolate_values(row, 'NDRE1'), axis=1)

# 保存插值后的数据到新的CSV文件
output_file_path = r"D:\0-code\AGRS_time_series-main\data\Hetian_TRAIN\experiment_3（8_2_0）\val_modified_interpolated.csv"
df.to_csv(output_file_path, index=False)

print(f"插值完成，结果已保存到 {output_file_path}")