import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

# i = [0,1,3,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,57,58,59,60,62,63,64,67,68]
i = [0]

for file_index in i:
    # 输入文件路径
    shp_file = r"C:\zyh\ktywork\20250703sichuan_agriculture\luorun.shp"  # 格网
    csv_file = fr"D:\0-code\AGRS_time_series-main\zyh\luorun_test.csv"  # 要修改label的csv
    txt_file = fr"D:\0-code\AGRS_time_series-main\predict_results\classification_WeiganFarmland_TimesNet_UEA_ftM_sl16_ll48_pl0_dm32_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0\predictions_and_probsLuorun.txt"  # 预测的txt
    val_csv_file = r"D:\0-code\AGRS_time_series-main\zyh\luorun_val.csv" # 加入样本像素
    train_csv_file = r"D:\0-code\AGRS_time_series-main\zyh\luorun_train.csv"  # 加入样本像素
    output_folder = r'D:\0-code\AGRS_time_series-main\tif_results'  # 输出栅格文件的文件夹

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取shp文件
    gdf = gpd.read_file(shp_file)

    # 读取csv文件
    csv_df = pd.read_csv(csv_file)
    val_df = pd.read_csv(val_csv_file)
    train_df = pd.read_csv(train_csv_file)

    # 读取txt文件
    with open(txt_file, 'r') as f:
        txt_data = f.readlines()

    # 将txt数据赋值给csv的cropcode列
    csv_df['cropcode'] = [int(line.strip()) for line in txt_data]

    # 筛选出val和train中gridcode一致的行
    val_df = val_df[val_df['gridcode'].isin(csv_df['gridcode'])]
    train_df = train_df[train_df['gridcode'].isin(csv_df['gridcode'])]

    # 将val和train的数据加入到csv_df中
    csv_df = pd.concat([csv_df, val_df, train_df], ignore_index=True)

    # 保存更新后的csv文件
    # csv_df.to_csv('updated_test_20_modified.csv', index=False)

    # 遍历每个gridcode
    for gridcode in csv_df['gridcode'].unique():
        # 筛选出当前gridcode对应的csv数据
        grid_csv_df = csv_df[csv_df['gridcode'] == gridcode]

        # 找到对应的shp几何范围
        grid_geometry = gdf[gdf['gridcode'] == gridcode].geometry.iloc[0]

        # 获取矢量范围的边界
        minx, miny, maxx, maxy = grid_geometry.bounds

        # 提取最大行号和列号
        max_row = grid_csv_df['row_col'].apply(lambda x: int(x.split('_')[0])).max()
        max_col = grid_csv_df['row_col'].apply(lambda x: int(x.split('_')[1])).max()

        # 创建栅格的仿射变换参数
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width=max_col, height=max_row)

        # 创建空的栅格数组
        raster_array = np.zeros((max_row, max_col), dtype=np.uint8)

        # 将cropcode值赋给栅格数组
        for index, row in grid_csv_df.iterrows():
            row_number, col_number = map(int, row['row_col'].split('_'))
            raster_array[row_number - 1, col_number - 1] = row['cropcode']

        # 保存栅格文件
        with rasterio.open(
                os.path.join(output_folder, f'gridcode_{file_index}.tif'),
                'w',
                driver='GTiff',
                height=max_row,
                width=max_col,
                count=1,
                dtype=raster_array.dtype,
                # crs='+proj=latlong',
                crs='+proj=utm +zone=44 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',  # 指定WGS 1984 / UTM zone 44N
                transform=transform,
        ) as dst:
            dst.write(raster_array, 1)

    print(f"已完成 i={file_index}")