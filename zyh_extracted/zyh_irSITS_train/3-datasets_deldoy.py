import csv

def delete_columns_and_rows_by_doy(csv_file_path, output_csv_file_path, doy_to_remove):
    # 读取CSV文件
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # 读取表头

        # 找到需要删除的列的索引
        columns_to_remove = []
        for doy in doy_to_remove:
            ndvi_col = f'NDVI_{doy}'
            ndre1_col = f'NDRE1_{doy}'
            if ndvi_col in header:
                columns_to_remove.append(header.index(ndvi_col))
            if ndre1_col in header:
                columns_to_remove.append(header.index(ndre1_col))

        # 创建一个新的表头，删除指定的列
        new_header = [col for i, col in enumerate(header) if i not in columns_to_remove]

        # 重新整理列顺序：gridcode, row_col, cropcode, objId, NDVI_doy1, NDVI_doy2, ..., NDRE1_doy1, NDRE1_doy2, ...
        ndvi_columns = sorted([col for col in new_header if col.startswith('NDVI_')], key=lambda x: int(x.split('_')[1]))
        ndre1_columns = sorted([col for col in new_header if col.startswith('NDRE1_')], key=lambda x: int(x.split('_')[1]))
        final_header = ['gridcode', 'row_col', 'cropcode', 'objId'] + ndvi_columns + ndre1_columns

        # 写入新的CSV文件
        with open(output_csv_file_path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(final_header)  # 写入新的表头

            # 遍历每一行数据，删除指定的列，并检查是否所有特征值都为None
            for row in reader:
                # 删除指定的列
                new_row = [col for i, col in enumerate(row) if i not in columns_to_remove]

                # 检查从第四个数据开始的所有特征值是否都为None
                if all(value == '' for value in new_row[3:]):
                    continue  # 如果所有特征值都为None，则跳过这一行

                # 重新整理列顺序
                gridcode, row_col = new_row[0].split('_', 1)  # 分解row_col为gridcode和row_col
                final_row = [gridcode, row_col, new_row[1], new_row[2]]
                for col in ndvi_columns:
                    final_row.append(new_row[new_header.index(col)])
                for col in ndre1_columns:
                    final_row.append(new_row[new_header.index(col)])

                # 写入修改后的行
                writer.writerow(final_row)

    print(f"列和行已成功删除，并且表格已整理，保存到 {output_csv_file_path}")

# 用户输入的doy较少的值的列表--需要删除的
doy_to_remove = [168, 153, 158, 258, 323, 8, 0, 20, 25, 65, 70, 80, 85, 115, 125, 130, 165, 180, 185, 190, 210, 215, 220, 255, 260, 265, 275, 280, 285, 295, 300, 305, 310, 315, 335, 340, 348, 353, 358]

# 输入和输出文件路径
csv_file_path = r"D:\0-code\AGRS_time_series-main\data\Hetian_TRAIN\experiment_3（8_2_0）\train.csv"  # 替换为你的CSV文件路径
output_csv_file_path = r"D:\0-code\AGRS_time_series-main\data\Hetian_TRAIN\experiment_3（8_2_0）\train_modified.csv"  # 替换为你想要保存的修改后的CSV文件路径

# 调用函数
delete_columns_and_rows_by_doy(csv_file_path, output_csv_file_path, doy_to_remove)