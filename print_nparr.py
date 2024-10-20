import numpy as np

# 指定要加载的 .npy 文件的路径
file_path_1 = 'data/1/csi_data_COM8_all_1.csv_amp_and_phase.npy'

# 使用 numpy.load() 函数加载 .npy 文件
data = np.load(file_path_1)

# 打印加载的数据
print(data)

# 如果你想检查数组的形状、类型等信息，可以使用以下语句：
print("Shape of the array:", data.shape)
print("Data type of the array:", data.dtype)

# 指定要加载的 .npy 文件的路径
file_path_1 = 'data/1/csi_data_COM8_all_1.csv_amp.npy'

# 使用 numpy.load() 函数加载 .npy 文件
data = np.load(file_path_1)

# 打印加载的数据
print(data)

# 如果你想检查数组的形状、类型等信息，可以使用以下语句：
print("Shape of the array:", data.shape)
print("Data type of the array:", data.dtype)

# 指定要加载的 .npy 文件的路径
file_path_1 = 'data/1/csi_data_COM8_all_1.csv_phase.npy'

# 使用 numpy.load() 函数加载 .npy 文件
data = np.load(file_path_1)

# 打印加载的数据
print(data)

# 如果你想检查数组的形状、类型等信息，可以使用以下语句：
print("Shape of the array:", data.shape)
print("Data type of the array:", data.dtype)