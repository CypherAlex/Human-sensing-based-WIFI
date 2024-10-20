import ast

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
from sympy.physics.control.control_plots import matplotlib

matplotlib.rc("font", family='MicroSoft YaHei', weight="bold")


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


def process_data(data):
    if data.shape[0] == 1501 and data.shape[1] == 52:
        #hampel滤波
        hampel_data = data
        for i in range(data.shape[1]):
            # 对于每一列，进行hampel滤波
            hampel_data[:, i] = hampel_filter(data[:, i])

        #低通滤波
        # 参数设置
        fs = 200  # 采样频率，根据实际数据设定
        cutoff_freq = 5  # 截止频率，这里设定为5Hz
        order = 6  # 滤波器阶数
        filtered_data = hampel_data
        for i in range(data.shape[1]):
            # 对于每一列，进行hampel滤波
            filtered_data[:, i] = apply_butterworth_filter(hampel_data[:, i], cutoff_freq, fs, order)

        # 初始化PCA模型，设置降维后的主成分数目为5
        pca = PCA(n_components=3)

        # 对数据进行PCA降维
        X_pca = pca.fit_transform(filtered_data)
        return X_pca
