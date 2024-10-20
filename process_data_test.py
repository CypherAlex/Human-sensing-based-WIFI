import ast

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
from statsmodels.sandbox.distributions.examples.ex_mvelliptical import fig
from sympy.physics.control.control_plots import matplotlib

matplotlib.rc("font", family='MicroSoft YaHei')
plt.rcParams['font.size'] = 30

def str_to_complex_list(s):
    # 将字符串转换为复数列表
    real_imag_pairs = ast.literal_eval(s.replace('[', '').replace(']', ''))
    return real_imag_pairs


def hampel_filter(data, k=1.4826, w=5):
    """
    应用Hampel滤波器去除数据中的异常值。

    参数:
    data - 1维numpy数组，需要过滤的数据。
    k - 异常值判断的系数，默认为1.4826，适用于标准正态分布。
    w - 窗口大小，必须为奇数，默认为5。
    """
    if w % 2 == 0:
        raise ValueError("窗口大小w必须是奇数")

    n = len(data)
    indices = np.arange(n)
    filtered_data = data.copy()

    for i in range(w // 2, n - w // 2):
        # 计算局部中位数
        x_median = np.median(data[i - w // 2:i + w // 2 + 1])

        # 计算绝对偏差的中位数
        mad = np.median(np.abs(data[i - w // 2:i + w // 2 + 1] - x_median))

        # 判断异常值
        threshold = k * mad

        # 替换异常值
        if np.abs(data[i] - x_median) > threshold:
            filtered_data[i] = x_median

    return filtered_data


def butter_lowpass(cutoff, fs, order=6):
    """
    设计一个巴特沃斯低通滤波器。

    参数:
    - cutoff: 截止频率
    - fs: 采样频率
    - order: 滤波器阶数
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_butterworth_filter(data, cutoff, fs, order):
    """
    应用巴特沃斯低通滤波器到数据上。

    参数:
    - data: 数据数组
    - cutoff: 截止频率
    - fs: 采样频率
    - order: 滤波器阶数
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


new_list = []
# 读取CSV文件
csv_path = ('data/circle1.csv')  # 请替换为你的CSV文件路径
data = pd.read_csv(csv_path)

data_list_amp = list(data['amp'])
# print(data_list_amp)
for data in data_list_amp:
    new_data = str_to_complex_list(data)
    new_data = new_data[0:52]
    new_list.append(new_data)

data = np.array(new_list)

print(data)
print(data.shape)
# 使用colormap获取不同颜色
cmap = matplotlib.colormaps['rainbow']  # 获取52种颜色的colormap以供更细致的映射
cmap = cmap.resampled(52)
plt.figure(figsize=(35, 12))

for i in range(data.shape[1]):
    # 对于每一列，绘制一条折线
    plt.plot(data[501:1000, i], color=cmap(i), label=f'Feature {i + 1}')

# 添加图例，但因为颜色太多，可能不适合全部显示，这里作为示例保留
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 添加标题和坐标轴标签
plt.title('原始数据52 条子载波中500个数据点的特征值')
plt.xlabel('数据包序号')
plt.ylabel('特征值')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()

plt.figure(figsize=(35, 12))
hampel_data = data
for i in range(data.shape[1]):
    # 对于每一列，进行hampel滤波
    hampel_data[:, i] = hampel_filter(data[:, i])
for i in range(hampel_data.shape[1]):
    plt.plot(hampel_data[501:1000, i], color=cmap(i), label=f'Feature {i + 1}')
fig.subplots_adjust(left=0, bottom=0.1, right=0.02, top=0.2)
# 添加标题和坐标轴标签
plt.title('经过hampel滤波处理后52条子载波中500个数据点的特征值')
plt.xlabel('数据包序号')
plt.ylabel('特征值')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()

# 假设参数
fs = 200  # 采样频率，根据实际数据设定
cutoff_freq = 5  # 截止频率，这里设定为5Hz

order = 6  # 滤波器阶数

plt.figure(figsize=(35, 12))
filtered_data = hampel_data
for i in range(data.shape[1]):
    # 对于每一列，进行hampel滤波
    filtered_data[:, i] = apply_butterworth_filter(hampel_data[:, i], cutoff_freq, fs, order)
for i in range(filtered_data.shape[1]):
    plt.plot(filtered_data[501:1000, i], color=cmap(i), label=f'Feature {i + 1}')
fig.subplots_adjust(left=0, bottom=0.1, right=0.02, top=0.2)
# 添加标题和坐标轴标签
plt.title('经过低通滤波处理后52条子载波中500个数据点的特征值')
plt.xlabel('数据包序号')
plt.ylabel('特征值')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()

# 初始化PCA模型，设置降维后的主成分数目为5
pca = PCA(n_components=5)

# 对数据进行PCA降维
X_pca = pca.fit_transform(filtered_data)

# 打印每个主成分所解释的方差百分比
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio (each component):")
for i, var in enumerate(explained_variance_ratio, start=1):
    print(f"PC {i}: {var * 100:.2f}%")
cmap = cmap.resampled(5)
# 输出降维后的数据形状验证
print("\nShape of transformed data:", X_pca.shape)
plt.figure(figsize=(35, 12))
for i in range(X_pca.shape[1]):
    plt.plot(X_pca[501:1000, i], color=cmap(i), label=f'Feature {i + 1}')
fig.subplots_adjust(left=0, bottom=0.1, right=0.02, top=0.2)
# 添加标题和坐标轴标签
plt.title('经过PCA处理后5条主成分中500个数据点的特征值')
plt.xlabel('数据包序号')
plt.ylabel('特征值')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
