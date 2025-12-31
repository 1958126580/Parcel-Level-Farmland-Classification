###对于有样本的格网
# 删除0行并且删除有地块编号的（这部分为样本集，避免重复）
def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.split(',')
            if '0,0,0,0' not in line and parts[2] == '0':
                outfile.write(line)

###对于没有样本的格网
# 删除0行
# def process_file(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             parts = line.split(',')
#             if '0,0,0,0' not in line:
#                 outfile.write(line)

i = [0]

# 遍历列表，处理每个文件
# for file_index in i:
#     input_file_path = fr"C:\zyh\data\fishnet_cty\{file_index}\pixel_deploy.txt"
#     output_file_path = fr"D:\xinjiang\hetian\sample_depoly_txt\{file_index}\pixel_deploy.txt"
#     process_file(input_file_path, output_file_path)
#     print(f"已完成 i={file_index}")

for file_index in i:
    input_file_path = r"C:\zyh\ktywork\20250703sichuan_agriculture\sichuan\time_series\luorun\cty\0\pixel_deploy.txt"
    output_file_path = r"C:\zyh\ktywork\20250703sichuan_agriculture\sichuan\time_series\luorun\cty\0\pixel_deploydel0.txt"
    process_file(input_file_path, output_file_path)
    print(f"已完成 i={file_index}")
