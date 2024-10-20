import ast
import os
import numpy as np
import pandas as pd

from process_data import process_data


def str_to_list(s):
    # 将字符串转换数字
    real_imag_pairs = ast.literal_eval(s.replace('[', '').replace(']', ''))
    return real_imag_pairs

new_list = []

dir_path = "C:/Users/18371/Desktop/data/"
dis_list = os.listdir(dir_path)
for first_dir in dis_list:
    file_list = os.listdir(dir_path + first_dir)
    for file in file_list:
        if file.endswith(".csv"):
            new_list = []
            data = pd.read_csv(dir_path + first_dir + "/" + file)
            data_amp_list = list(data['amp'])

            for data in data_amp_list:
                data = str_to_list(data)[0:52]
                new_list.append(data)
            data_amp_array = np.array(new_list)
            if data_amp_array.shape[0] == 1501 and data_amp_array.shape[1] == 52:
                # 对数据进行预处理
                data_amp_array = process_data(data_amp_array)
                output_file_path = f"{file}_amp.npy"

                # 步骤 4: 将NumPy数组保存到新的文件中
                np.save(dir_path+"/"+first_dir+"/"+output_file_path, data_amp_array)

                print(f"数据已成功保存到 {output_file_path}")
for first_dir in dis_list:
    file_list = os.listdir(dir_path + first_dir)
    for file in file_list:
        if file.endswith(".csv"):
            new_list = []
            data = pd.read_csv(dir_path + first_dir + "/" + file)
            data_phase_list = list(data['phase'])

            for data in data_phase_list:
                data = str_to_list(data)[0:52]
                new_list.append(data)
            data_phase_array = np.array(new_list)
            if data_phase_array.shape[0] == 1501 and data_phase_array.shape[1] == 52:
                output_file_path = f"{file}_phase.npy"
                # 对数据进行预处理
                data_phase_array = process_data(data_phase_array)
                # 步骤 4: 将NumPy数组保存到新的文件中
                np.save(dir_path+"/"+first_dir+"/"+output_file_path, data_phase_array)

                print(f"数据已成功保存到 {output_file_path}")