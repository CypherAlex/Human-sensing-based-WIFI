import ast
import json
import os
import random
import numpy as np
import pandas as pd
from keras import Sequential
from scikeras.wrappers import KerasClassifier
from keras.src.layers import Bidirectional, BatchNormalization
from keras.src.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import L2

dir_path = "C:/Users/18371/Desktop/ModelBuild/data/"
dis_list = os.listdir(dir_path)

# 初始化数据和标签数组
data = []
label_encoder = LabelEncoder()
labels = []

for idx, first_dir in enumerate(dis_list):
    file_list = os.listdir(os.path.join(dir_path, first_dir))
    for file in file_list:
        if file.endswith("_amp_and_phase.npy"):
            labels.append(first_dir)
            file_path = os.path.join(dir_path, first_dir, file)
            new_data = np.load(file_path)
            #new_data = new_data.reshape(-1, 5)  # 假设每行代表一个时间步的5个特征
            data.append(new_data)

data = np.array(data)
labels = label_encoder.fit_transform(labels)

# 数据预处理
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 5)).reshape(data.shape)  # 标准化每个特征
print(data.shape)
print(labels.shape)
# 划分数据集
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.15, random_state=42)

# 转换标签为one-hot编码
num_classes = len(np.unique(labels))
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


# 构建LSTM模型
def build_lstm_model(optimizer='adam'):
    model = Sequential([
        # 第一层双向LSTM，包含dropout和recurrent_dropout参数
        Bidirectional(
            LSTM(64,
                 return_sequences=True,
                 input_shape=(1501, 10),
                 dropout=0.2,
                 recurrent_dropout=0.2,
                 kernel_regularizer=L2(0.001)),  # 添加L2正则化)
        ),
        #BatchNormalization(),  # 添加Batch Normalization
        Dropout(0.2),

        # 第二层双向LSTM，同样包含dropout和不返回序列
        Bidirectional(
            LSTM(32,
                 dropout=0.2,
                 recurrent_dropout=0.2)
        ),
        BatchNormalization(),  # 添加Batch Normalization
        Dropout(0.2),

        # 输出层，使用softmax激活函数
        Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    if optimizer == 'adam':
        opt = Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=0.01)
    else:
        raise ValueError("Invalid optimizer, choose from 'adam' or 'sgd'.")

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 使用SGD优化器构建模型
model_sgd = build_lstm_model(optimizer='sgd')

# 训练模型
history_sgd = model_sgd.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# 评估模型
loss_sgd, accuracy_sgd = model_sgd.evaluate(test_data, test_labels)
print(f'Test accuracy using SGD optimizer: {accuracy_sgd}')