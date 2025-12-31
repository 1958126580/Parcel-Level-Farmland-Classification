import numpy as np
from datetime import datetime
import os
import time

def date_to_doy(date):
    base_date = 20240101
    date_str = str(date)
    base_str = str(base_date)
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    base_obj = datetime.strptime(base_str, "%Y%m%d")
    delta = date_obj - base_obj
    return delta.days

def calculate_ndvi(red, nir):
    if red == 0 or nir == 0:
        return np.nan  # 如果 red 或 nir 为 0，返回 NaN
    return np.around((nir - red) / (nir + red), decimals=4)

def calculate_ndre1(b5, b6):
    if b5 == 0 or b6 == 0:
        return np.nan  # 如果 b5 或 b6 为 0，返回 NaN
    return np.around((b6 - b5) / (b6 + b5), decimals=4)

def process_file(input_file, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()  # 记录开始时间
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("row_col,cropcode,objId,doy,NDVI,NDRE1\n")
        for line in infile:
            elements = line.strip().split(',')
            row_col = elements[0]
            cropcode = elements[1]
            objId = elements[2]
            features = []
            for i in range(3, len(elements), 5):
                date = int(elements[i])
                red = float(elements[i + 1])
                b5 = float(elements[i + 2])
                b6 = float(elements[i + 3])
                nir = float(elements[i + 4])
                doy = date_to_doy(date)
                ndvi = calculate_ndvi(red, nir)
                ndre1 = calculate_ndre1(b5, b6)
                if not np.isnan(ndvi) and not np.isnan(ndre1):
                    features.append(f"{doy},{ndvi},{ndre1}")
            if features:  # 如果有有效的特征值，写入文件
                outfile.write(f"{row_col},{cropcode},{objId},{','.join(features)}\n")
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算用时
    print(f"处理文件所用时间：{elapsed_time:.2f} 秒")  # 打印用时

# i = [0,1,3,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,57,58,59,60,62,63,64,67,68]
i = [50,53,59]

for file_index in i:
    input_file = fr'D:\xinjiang\hetian\sample_depoly_txt\{file_index}\pixel_deploy.txt'  # 输入文件名
    output_file = fr'D:\xinjiang\hetian\deploy_test_txt\{file_index}\pixel_deploy.txt'  # 输出文件名
    process_file(input_file, output_file)
    print(f"已完成 i={file_index}")
# input_file = r"C:\Users\aircas\Desktop\test.txt"  # 输入文件名
# output_file = r"C:\Users\aircas\Desktop\output.txt"  # 输出文件名
# process_file(input_file, output_file)