import os

def save_npy_directory_to_txt(directory, output_file):
    # 获取指定目录下所有文件和文件夹的路径
    paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    # 筛选出.npy文件
    npy_files = [path for path in paths if path.endswith('.npy')]
    # 将.npy文件的路径写入文本文件
    with open(output_file, 'w') as f:
        for file in npy_files:
            f.write(file + '\n')

# 用法示例
save_npy_directory_to_txt('somedata', 'output.txt')
