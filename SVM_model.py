import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

dir_path = "C:/Users/18371/Desktop/ModelBuild/data/"
dis_list = os.listdir(dir_path)

# 初始化数据和标签数组
data = np.array([])
label_encoder = LabelEncoder()  # 用于将字符串标签转换为整数
labels = []

for idx, first_dir in enumerate(dis_list):
    file_list = os.listdir(os.path.join(dir_path, first_dir))
    for file in file_list:
        if file.endswith("_amp_and_phase.npy"):
            labels.append(first_dir)  # 收集标签
            file_path = os.path.join(dir_path, first_dir, file)
            new_data = np.load(file_path)
            new_data = np.reshape(new_data, (-1))  # 假设每个文件只包含一维数据，直接展平
            if data.size == 0:
                data = new_data
            else:
                data = np.vstack((data, new_data))
                print("Successfully loaded "
                      ""+file)

# 将标签转换为整数
labels = label_encoder.fit_transform(labels)

# 数据标准化
X_scaler = StandardScaler()
data_scaled = X_scaler.fit_transform(data)

# 分割数据集
train_data, test_data, train_label, test_label = train_test_split(data_scaled, labels, random_state=1, train_size=0.7, test_size=0.3)

# 使用GridSearchCV进行参数调优
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(svm.SVC(decision_function_shape='ovo'), param_grid, cv=5)
grid_search.fit(train_data, train_label)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最优参数评估模型
best_model = grid_search.best_estimator_
train_score = best_model.score(train_data, train_label)
test_score = best_model.score(test_data, test_label)

# print("训练集准确率: ", train_score)
print("测试集准确率: ", test_score)