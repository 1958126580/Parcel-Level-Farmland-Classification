### 将格式整理为转为tsv格式的csv ###

import pandas as pd

# i = [0,1,3,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,57,58,59,60,62,63,64,67,68]
i = [50,53,59]

for file_index in i:
    # 读取CSV文件（anaconda处理得到的结果）
    input_file = fr"D:\0-code\AGRS_time_series-main\data\Hetian_TEST\test_{file_index}_modified.csv"
    output_file = fr"D:\0-code\AGRS_time_series-main\data\Hetian_TEST\Hetian_{file_index}_TEST.ts"

    # 读取CSV文件，跳过第一行（列标题）
    df = pd.read_csv(input_file, header=None, skiprows=1)

    # 删除指定列
    # 删除gridcode、row_col、objId列（它们是第0、1、3列）  只留label列
    columns_to_drop = [0, 1, 3]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # 将cropcode列（第2列）移动到最右边
    # 首先获取所有列的索引，除了cropcode列
    columns = [col for col in df.columns if col != 2]
    # 将cropcode列添加到列索引列表的最后
    columns.append(2)
    # 重新排列DataFrame的列顺序
    df = df[columns]

    # 将特征值转换为字符串，并用逗号连接，然后与cropcode用冒号连接
    # 提取特征值列和cropcode列
    features = df.iloc[:, :-1].astype(str)
    cropcode = df.iloc[:, -1].astype(str)

    # 将特征值列转换为逗号分隔的字符串
    features_str = features.apply(lambda row: ','.join(row), axis=1)

    # 将特征值字符串与cropcode用冒号连接
    formatted_data = features_str + ':' + cropcode

    # 将格式化后的数据保存为TSV文件
    with open(output_file, 'w') as file:
        # 添加指定内容到TSV文件的最前面
        header_content = """#A set of eight simple gestures generated from accelerometers. The data consists of the X,Y,Z coordinates of each motion. Each series is 315 long. We have
#First described in [1]. 
#
#J. Liu, Z. Wang, L. Zhong, J. Wickramasuriya and V. Vasudevan, "uWave: Accelerometer-based personalized gesture recognition and its applications,"
#2009 IEEE International Conference on Pervasive Computing and Communications, Galveston, TX, 2009, pp. 1-9.
@problemName WeiganFarmland
@timeStamps false
@missing false
@univariate false
@dimensions 2
@equalLength true
@seriesLength 30
@classLabel true 1 2 3 4 5 6 7 8
@data
"""
        file.write(header_content)

        # 写入格式化后的数据
        for line in formatted_data:
            file.write(line + '\n')

    print(f"已完成 i={file_index}")