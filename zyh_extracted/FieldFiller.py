import argparse
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import mapping
from concurrent.futures import ProcessPoolExecutor
import time  # 导入time模块

field_input_path = r"D:\xinjiang\hetian\6-linetopolygon\field_new.shp"
classification_tif_path = r"D:\0-code\AGRS_time_series-main\tif_results\hetian\predict.tif"
field_output_path = r"D:\xinjiang\hetian\6-linetopolygon\field_new_filler.shp"

def count_typecodes_in_polygon(polygon, band, transform, out_shape):
    """
    统计多边形中出现次数最多的栅格类型代码，如果多边形没有覆盖任何像素，
    则根据其质心所在的像素确定类型代码。
    """
    geoms = [mapping(polygon)]
    burned = features.rasterize(geoms, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    pixels = band[burned > 0]

    if pixels.size == 0:
        # 如果没有覆盖的像素，获取多边形质心所在的像素
        centroid = polygon.centroid
        row, col = rasterio.transform.rowcol(transform, centroid.x, centroid.y)
        if 0 <= row < out_shape[0] and 0 <= col < out_shape[1]:
            return band[row, col]
        else:
            return -1  # 如果质心超出栅格范围，返回-1

    typecode_counts = np.bincount(pixels.astype(int))  # 统计每个类型代码的像素数量
    major_typecode = np.argmax(typecode_counts)  # 找到数量最多的类型代码

    return major_typecode

def process_polygon(row, band, transform, out_shape):
    """
    处理单个多边形，返回Fieldcode和Typecode
    """
    polygon = row['geometry']  # 直接使用GeoDataFrame中的几何对象
    major_typecode = count_typecodes_in_polygon(polygon, band, transform, out_shape)
    return row['Fieldcode'], major_typecode

def main():
    start_time = time.time()  # 记录开始时间

    # 打开栅格文件
    with rasterio.open(classification_tif_path) as src:
        band = src.read(1)  # 读取第一波段
        transform = src.transform  # 获取仿射变换参数
        out_shape = band.shape  # 获取输出图像的尺寸

    # 读取.shp文件
    gdf = gpd.read_file(field_input_path)

    # 检查是否存在Fieldcode字段，如果不存在则创建
    if 'Fieldcode' not in gdf.columns:
        gdf['Fieldcode'] = gdf.index.astype(np.int64) + 1

    # 使用多进程并行处理
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_polygon, row, band, transform, out_shape) for _, row in gdf.iterrows()]
        results = [future.result() for future in futures]

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results, columns=['Fieldcode', 'Typecode'])

    # 合并结果到原始GeoDataFrame
    gdf_with_typecode = gdf.merge(results_df, on='Fieldcode', how='left')

    # 填充没有typecode的地块
    gdf_with_typecode['Typecode'] = gdf_with_typecode['Typecode'].fillna(-1)

    # 保存更新后的.shp文件
    gdf_with_typecode.to_file(field_output_path, driver='ESRI Shapefile')

    end_time = time.time()  # 记录结束时间
    print(f"程序运行完成，总耗时：{end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main()

# import argparse
# import rasterio
# import geopandas as gpd
# import numpy as np
# import pandas as pd
# from rasterio import features
# from shapely.geometry import shape, mapping
#
# field_input_path = r"D:\xinjiang\hetian\6-linetopolygon\25 end.shp"
# classification_tif_path = r"D:\0-code\AGRS_time_series-main\tif_results\predict.tif"
# field_output_path = r"D:\xinjiang\hetian\6-linetopolygon\25 end_fill.shp"
#
# # def parse_arguments():
# #     parser = argparse.ArgumentParser(description="Process input and output paths.")
# #     parser.add_argument("field_input_path", type=str, help="Path to the input SHP file")
# #     parser.add_argument("classification_tif_path", type=str, help="Path to the classification TIF file")
# #     parser.add_argument("field_output_path", type=str, help="Path to the output SHP file")
# #     return parser.parse_args()
#
# def count_typecodes_in_polygon(polygon, band, transform, out_shape):
#     """
#     统计多边形中出现次数最多的栅格类型代码
#     """
#     geoms = [mapping(polygon)]
#     burned = features.rasterize(geoms, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
#     pixels = band[burned > 0]
#
#     if pixels.size == 0:
#         return -1  # 如果没有覆盖的像素，返回-1
#
#     typecode_counts = np.bincount(pixels.astype(int))  # 统计每个类型代码的像素数量
#     major_typecode = np.argmax(typecode_counts)  # 找到数量最多的类型代码
#
#     return major_typecode
#
# def main():
#     # # 解析命令行参数
#     # args = parse_arguments()
#     #
#     # # 从命令行参数中获取路径
#     # field_input_path = args.field_input_path
#     # classification_tif_path = args.classification_tif_path
#     # field_output_path = args.field_output_path
#
#     # 打开栅格文件
#     with rasterio.open(classification_tif_path) as src:
#         band = src.read(1)  # 读取第一波段
#         transform = src.transform  # 获取仿射变换参数
#         out_shape = band.shape  # 获取输出图像的尺寸
#
#     # 读取.shp文件
#     gdf = gpd.read_file(field_input_path)
#
#     # 检查是否存在Fieldcode字段，如果不存在则创建
#     if 'Fieldcode' not in gdf.columns:
#         # 创建Fieldcode字段，值为FID+1（FID从0开始）
#         gdf['Fieldcode'] = gdf.index.astype(np.int64) + 1
#
#     # 初始化一个空的DataFrame来存储结果
#     results = pd.DataFrame(columns=['Fieldcode', 'Typecode'])
#
#     # 遍历每个地块
#     for index, row in gdf.iterrows():
#         polygon = shape(row['geometry'])  # 将地块转换为shapely几何对象
#         major_typecode = count_typecodes_in_polygon(polygon, band, transform, out_shape)
#
#         # 检查是否获得了有效的typecode
#         if major_typecode != -1:
#             # 创建一个新的DataFrame，只包含一行数据
#             new_row = pd.DataFrame({'Fieldcode': [row['Fieldcode']], 'Typecode': [major_typecode]})
#             # 使用concat函数将新行添加到results DataFrame中
#             results = pd.concat([results, new_row], ignore_index=True)
#
#     # 将结果合并到原始的gdf中，注意只合并那些有结果的行
#     gdf_with_typecode = gdf.merge(results[['Fieldcode', 'Typecode']], on='Fieldcode', how='left')
#
#     # 填充那些没有typecode的地块，使用-1作为默认值
#     gdf_with_typecode['Typecode'] = gdf_with_typecode['Typecode'].fillna(-1)
#
#     # 保存更新后的.shp文件
#     gdf_with_typecode.to_file(field_output_path, driver='ESRI Shapefile')
#
# if __name__ == "__main__":
#     main()