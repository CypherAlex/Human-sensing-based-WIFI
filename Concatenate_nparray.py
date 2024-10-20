import os
import numpy as np


def merge_npy_files_in_subfolders(directory):
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        # 检查是否为最底层目录（没有子目录）
        if not dirs:
            # 创建一个字典来存储前缀和对应的文件列表
            prefix_to_files = {}

            # 对于当前目录下的每个文件
            for file in files:
                if file.endswith('.npy'):
                    # 提取文件前缀（假设前缀由'_amp'或'_phase'结尾）
                    prefix = os.path.splitext(file)[0].rsplit('_', 1)[0]

                    # 如果前缀不在字典中，添加一个新的键值对
                    if prefix not in prefix_to_files:
                        prefix_to_files[prefix] = []

                    # 将文件添加到对应前缀的列表中
                    prefix_to_files[prefix].append(os.path.join(root, file))

            # 遍历字典，合并具有相同前缀的文件
            for prefix, file_list in prefix_to_files.items():
                # 确保至少有两个文件才能进行合并
                if len(file_list) >= 2:
                    # 读取第一个文件的数据
                    data = np.load(file_list[0])

                    # 合并所有其他文件数据
                    for file_path in file_list[1:]:
                        data = np.hstack((data, np.load(file_path)))

                    # 生成新的文件名
                    new_file_name = f"{prefix}_amp_and_phase.npy"

                    # 保存合并后的数据到新文件
                    np.save(os.path.join(root, new_file_name), data)
                    print(f"Merged {len(file_list)} files into {new_file_name}")


# 调用函数，传入你的目录路径
merge_npy_files_in_subfolders('C:/Users/18371/Desktop/data1/')