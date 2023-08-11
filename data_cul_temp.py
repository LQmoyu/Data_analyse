'''
2023/7/20

'''
import itertools
import scipy.io as scio
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from pyts.image import GramianAngularField
from matplotlib import font_manager  # 导入字体管理模块
import os
from filterpy.kalman import KalmanFilter
from scipy.ndimage import gaussian_filter, median_filter
import pywt
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother
from tsmoothie.utils_func import sim_seasonal_data
from tsmoothie.smoother import DecomposeSmoother
from sklearn.metrics import confusion_matrix
import hdf5storage

def cul_data(data):
    """
    cul_data处理

    data:  原始数据

    return: 返回值是处理后的结果
    """
    ## 阵元归一
    divisor = data[:,0]

    divisor = divisor[:, np.newaxis]
    # print(divisor.shape)

    temp_data = data / divisor
    # 特征工程 
    ## tsmoothie 
    ## 修改行数
    # sm_data = temp_data[1000:2001, :].T
    sm_data = temp_data.T  

    # operate smoothing
    #参数整定
    smoother = DecomposeSmoother(smooth_type='lowess', periods=50,
                                smooth_fraction=0.2)
    smoother.smooth(sm_data)

    low, up = smoother.get_intervals('sigma_interval')

    # return smoother.smooth_data # 处理后的数据
    return sm_data # 原始数据

def add_label(data, label):
    '''
    add_label:

    data:原始数据

    label: 标签
    
    return: 
    '''
    news_row = np.array([label] *data.shape[1])

    arr = np.vstack((data, news_row))
    
    return arr

def merge_data(data):
    '''
    data: 传入列表数据
    return: 合并后的数组 arr
    '''
    temp_data = data[0]

    for i in range(1, len(data)):
        merge_arr = np.concatenate((temp_data, data[i]), axis=1)
        temp_data = merge_arr

    return temp_data

def handle_data():
    '''
        handle_data:首先调用，对数据进行拆解处理
        return: 整个数据集 滤波后的数据
    '''
    path = ['../data/MM_muti_0.mat', '../data/MM_muti_04.mat', '../data/MM_muti_07.mat']
    all_data_test = []
    for i in path:
        mat_data = hdf5storage.loadmat(i)
        temp_data = mat_data['MM_muti_F1_no_nor'][0]
        
        temp_data = cul_data(path)
        all_data_test.append(add_label(temp_data, i))

    return all_data_test

def cnf_matrix_plotter(cm, classes, accuracy, cmap=plt.cm.Blues):
    """
    传入混淆矩阵和标签名称，绘制混淆矩阵
    """
    plt.figure(figsize=(5, 5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar() # 色条
    tick_marks = np.arange(classes)

    plt.title('confusion_matrix-rf-{:.3f}.pdf'.format(accuracy), fontsize=12)
    # plt.title('confusion_matrix-rf_no_filter-{:.3f}.pdf'.format(accuracy), fontsize=12)

    plt.colorbar()
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tick_params(labelsize=10)  # 设置类别文字大小
    plt.xticks(tick_marks)  # x轴文字旋转
    plt.yticks(tick_marks)

    # 写数�?
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=8)

    plt.tight_layout()

    plt.savefig('confusion_matrix-rf-{:.3f}.pdf'.format(accuracy), dpi=300)  # 保存图像

    # plt.savefig('confusion_matrix-rf_no_filter-{:.3f}.pdf'.format(accuracy), dpi=300)  # 保存图像
    plt.show()

