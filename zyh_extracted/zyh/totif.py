import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

i = [0]

for file_index in i:
    # 输入文件路径
    shp_file = r"C:\zyh\ktywork\20250703sichuan_agriculture\luorun.shp"# 格网
    csv_file = fr"D:\0-code\AGRS_time_series-main\zyh\luorun_test.csv"  # 要修改label的csv
    txt_file = fr"D:\0-code\AGRS_time_series-main\predict_results\classification_WeiganFarmland_TimesNet_UEA_ftM_sl14_ll48_pl0_dm32_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0\predictions_and_probsLuorun.txt" # 预测的txt
    val_csv_file = r"D:\0-code\AGRS_time_series-main\zyh\luorun_val.csv" # 加入样本像素
    train_csv_file = r"D:\0-code\AGRS_time_series-main\zyh\luorun_train.csv"  # 加入样本像素
    output_folder = r'D:\0-code\AGRS_time_series-main\tif_results'  # 输出栅格文件的文件夹

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取shp文件
    gdf = gpd.read_file(shp_file)

    print("矢量数据边界范围：", gdf.total_bounds)

    # 读取csv文件
    csv_df = pd.read_csv(csv_file)
    val_df = pd.read_csv(val_csv_file)
    train_df = pd.read_csv(train_csv_file)

    # 读取txt文件
    with open(txt_file, 'r') as f:
        txt_data = f.readlines()

    # 将txt数据赋值给csv的cropcode列
    csv_df['label'] = [int(line.strip()) for line in txt_data]

    # 将val和train的数据加入到csv_df中
    csv_df = pd.concat([csv_df, val_df, train_df], ignore_index=True)
    csv_df = pd.concat([csv_df], ignore_index=True)

    # 保存更新后的csv文件
    # csv_df.to_csv('luorun_modified.csv', index=False)

    # 获取矢量范围的边界
    minx, miny, maxx, maxy = gdf.total_bounds

    # 提取最大行号和列号
    max_row = csv_df['Row_Column'].apply(lambda x: int(x.split('_')[0])).max()
    max_col = csv_df['Row_Column'].apply(lambda x: int(x.split('_')[1])).max()
    print("最大行号：", max_row)
    print("最大列号：", max_col)

    # 创建栅格的仿射变换参数
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width=max_col, height=max_row)

    # 创建空的栅格数组
    raster_array = np.zeros((max_row, max_col), dtype=np.uint8)

    # 将cropcode值赋给栅格数组
    for index, row in csv_df.iterrows():
        row_number, col_number = map(int, row['Row_Column'].split('_'))
        raster_array[row_number - 1, col_number - 1] = row['label']

    # 保存栅格文件
    with rasterio.open(
            os.path.join(output_folder, f'luorun3.tif'),
            'w',
            driver='GTiff',
            height=max_row,
            width=max_col,
            count=1,
            dtype=raster_array.dtype,
            crs='EPSG:4326',  # 指定 WGS1984 地理坐标系
            transform=transform,  # 确保 transform 是地理坐标系的仿射变换参数
    ) as dst:
        dst.write(raster_array, 1)

    print(f"已完成 i={file_index}")

# import pandas as pd
# import rasterio
# from rasterio.transform import from_origin
# import numpy as np
#
# # 读取CSV表格
# csv_file = r"D:\0-code\AGRS_time_series-main\zyh\luorun_modified.csv"   # 替换为你的CSV文件路径
# df = pd.read_csv(csv_file)
#
# # 读取栅格文件以获取其元数据
# raster_file = r"C:\zyh\ktywork\20250703sichuan_agriculture\sichuan\time_series\luorun\cty\0\20240311012330924.tif"  # 替换为你的栅格文件路径
# with rasterio.open(raster_file) as src:
#     # 获取栅格的元数据
#     meta = src.meta.copy()
#     # 获取栅格的变换信息
#     transform = src.transform
#     # 获取栅格的坐标系
#     crs = src.crs
#
# # 根据Row_Column的行列号生成tif文件
# # 假设Row_Column的格式为"row_column"
# # 例如："1_2"表示第1行第2列
# # 你需要根据实际情况调整解析方式
# row_col = df['Row_Column'].str.split('_', expand=True)
# row_col.columns = ['row', 'column']
# row_col = row_col.astype(int)
#
# # 获取栅格的尺寸
# height = meta['height']
# width = meta['width']
#
# # 创建一个与栅格尺寸相同的数组
# label_array = np.zeros((height, width), dtype=np.int32)
#
# # 根据行列号和label填充数组
# for index, row in row_col.iterrows():
#     r = row['row']
#     c = row['column']
#     label = df.loc[index, 'label']
#     # 注意：这里假设行列号是从1开始的，如果你的行列号是从0开始的，请去掉减1
#     label_array[r - 1, c - 1] = label
#
# # 更新元数据
# meta.update(dtype=rasterio.int32, count=1)
#
# # 写入新的tif文件
# output_file = 'output.tif'  # 替换为你想要保存的文件路径
# with rasterio.open(output_file, 'w', **meta) as dst:
#     dst.write(label_array, 1)
#
# print(f"文件已保存为 {output_file}")