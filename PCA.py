
import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sympy.physics.control.control_plots import matplotlib
matplotlib.rc("font",family='MicroSoft YaHei',weight="bold")
def str_to_complex_list(s):
    # 将字符串转换为复数列表
    real_imag_pairs = ast.literal_eval(s.replace('[', '').replace(']', ''))
    return real_imag_pairs

new_list = []
path = "data/1/csi_data_COM8_all_1.csv_amp_and_phase.npy"
data = np.load(path)

x = data
print(x.shape)
# 初始化PCA模型，设置降维后的主成分数目为5
pca = PCA(n_components=10)

# 对数据进行PCA降维
X_pca = pca.fit_transform(x)

# 打印每个主成分所解释的方差百分比
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio (each component):")
for i, var in enumerate(explained_variance_ratio, start=1):
    print(f"PC {i}: {var * 100:.2f}%")
# 输出降维后的数据形状验证
print("\nShape of transformed data:", X_pca.shape)

plt.show()
