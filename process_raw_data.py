import pandas as pd
import os
import ast
import numpy as np


def str_to_complex_list(s):
    # 将字符串转换为复数列表
    real_imag_pairs = ast.literal_eval(s.replace('[', '').replace(']', ''))
    complex_list = [complex(real, imag) for real, imag in zip(real_imag_pairs[::2], real_imag_pairs[1::2])]
    return complex_list


def process_csv(file_path):
    # 读取CSV文件
    print("Processing:", file_path)
    df = pd.read_csv(file_path)
    # 判断是否有data列
    if 'data' in df.columns:
        # 将"data"列的字符串转换为复数列表
        df['complex_data'] = df['data'].apply(str_to_complex_list)
        df['amp'] = df['complex_data'].apply(lambda x: [round(np.abs(y), 4) for y in x])
        df['phase'] = df['complex_data'].apply(lambda x: [round(np.angle(y), 4) for y in x])
        # 保存处理后的数据（如果需要的话）
        df.to_csv(file_path, index=False)
        #print(df['amp'])
        #print(df['phase'])
    print("successfully Processed:", file_path)

# 假设您的文件夹路径如下，根据实际情况修改
dir_path = "C:/Users/18371/Desktop/data/"
dis_list = os.listdir(dir_path)
dic = {}
for first_dir in dis_list:
    file_list = os.listdir(dir_path + first_dir)
    for file in file_list:
        if file.endswith(".csv"):
            process_csv(dir_path + first_dir + "/" + file)
