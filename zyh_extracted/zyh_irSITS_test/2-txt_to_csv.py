import csv

def txt_to_csv(txt_file_path, csv_file_path):
    # 打开txt文件并读取数据
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    # 初始化一个集合来存储所有doy值（Day of Year，即一年中的第几天）
    all_doy = set()

    # 遍历每一行数据，提取doy值
    for line in lines:
        parts = line.strip().split(',')  # 按逗号分割每行数据
        # 提取doy值，从第4个元素开始，每隔3个元素提取一个doy值
        for i in range(3, len(parts), 3):
            try:
                doy = int(parts[i])  # 尝试将doy值转换为整数
                all_doy.add(doy)  # 将提取的doy值添加到集合中
            except ValueError:
                # 如果转换失败，跳过该值
                continue

    # 将doy值排序，确保顺序一致
    all_doy = sorted(list(all_doy))

    # 准备csv文件的表头
    # 表头包括row_col, cropcode, objId，以及每个doy对应的NDVI和NDRE1列
    header = ['row_col', 'cropcode', 'objId']
    for doy in all_doy:
        header.append(f'NDVI_{doy}')
        header.append(f'NDRE1_{doy}')

    # 打开csv文件并写入数据
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)  # 写入表头

        # 遍历每一行数据，转换为csv格式
        for line in lines:
            parts = line.strip().split(',')  # 按逗号分割每行数据
            row_col = parts[0]  # 格网号_行号_列号
            cropcode = parts[1]  # 作物类型
            objId = parts[2]  # 地块编号

            # 初始化一个列表来存储当前像素的csv行数据
            csv_row = [row_col, cropcode, objId]

            # 遍历所有doy值
            for doy in all_doy:
                # 检查当前doy是否存在于该像素的数据中
                if str(doy) in parts[3::3]:  # 检查doy是否在数据中
                    # 找到对应的索引
                    index = parts[3::3].index(str(doy))
                    ndvi = parts[3 + index * 3 + 1]  # 提取对应的NDVI值
                    ndre1 = parts[3 + index * 3 + 2]  # 提取对应的NDRE1值
                else:
                    ndvi = None  # 如果doy不存在，NDVI值为None
                    ndre1 = None  # 如果doy不存在，NDRE1值为None
                csv_row.extend([ndvi, ndre1])  # 将NDVI和NDRE1值添加到csv行数据中

            # 写入当前像素的csv行数据
            writer.writerow(csv_row)

    # 删除第二行
    delete_second_row(csv_file_path)

def delete_second_row(csv_file_path):
    # 读取CSV文件
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    # 删除第二行
    if len(rows) > 1:
        del rows[1]

    # 重新写入CSV文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)

# i = [0,1,3,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,57,58,59,60,62,63,64,67,68]
i = [59]

for file_index in i:
    txt_file_path = fr'D:\xinjiang\hetian\deploy_test_txt\{file_index}\pixel_deploy.txt'  # 输入文件名
    csv_file_path = fr'D:\0-code\AGRS_time_series-main\data\Hetian_TEST\test_{file_index}.csv'   # 输出文件名
    txt_to_csv(txt_file_path, csv_file_path)
    print(f"已完成 i={file_index}")
