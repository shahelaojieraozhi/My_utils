import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import transforms, datasets


def read_data(data_path):
    '''
    read original data, show and plot
    Feature和label的读取
    '''
    data = pd.read_excel(data_path)
    # 如果Feature和label在excel里面比较分散,先将他们聚集
    data = pd.concat([pd.DataFrame(data.iloc[:, 2]), pd.DataFrame(data.iloc[:, 6:9])], axis=1)
    print(data.head())
    features = data.iloc[:, 1:4]
    label = data.iloc[:, [0]]

    # 可以画出来大致看看长啥样
    # plt.plot(data.iloc[:,0])
    # plt.show()

    return features, label


"""模型的保存和使用"""

import joblib
import numpy as np


# 存一下模型
def model_zip(bp, model_path):
    joblib.dump(bp, model_path)  # 存储
    '''
    bp指某个模型，这里可以是bpnn
    比如：output_path = "./bp1_train_model.m"
    '''


# 模型的使用
def model_use(model_path, test_data):
    # 读取模型
    bp = joblib.load(model_path)  # 调用
    # 预测数据
    b = bp.predict(test_data)
    y_pre = np.array(b)  # 列表转数组
    return y_pre


from sklearn.preprocessing import MinMaxScaler


def normalization(features, label):
    '''
    normalization
    '''
    mm_x = MinMaxScaler()
    mm_y = MinMaxScaler()
    features = features.values
    data = mm_x.fit_transform(features)
    label = mm_y.fit_transform(label)
    return data, label, mm_y


