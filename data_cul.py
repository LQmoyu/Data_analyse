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

def cul_data(path):
    """
    cul_data处理

    path:  数据所在的文件路径

    return: 返回值是处理后的结果
    """
    num = int(path[-5])  # 读取文件命名
    f = h5py.File(path, 'r')
    if num == 0:
        element_name = 'MM_multi_MA_'
    else:
        element_name = 'MM_multi_MA_0'

    src_data = np.array(f[element_name + str(num)])

    ## 阵元归一
    divisor = src_data[:,0]

    divisor = divisor[:, np.newaxis]
    # print(divisor.shape)

    temp_data = src_data / divisor

    # 特征工程 
    ## tsmoothie 
    ## 修改行数
    sm_data = temp_data[1000:2001, :].T  

    # operate smoothing
    #参数整定
    smoother = DecomposeSmoother(smooth_type='lowess', periods=50,
                                smooth_fraction=0.2)
    smoother.smooth(sm_data)
    # plot the smoothed timeseries with intervals
    # plt.figure(figsize=(18,10))
    # generate intervals
    low, up = smoother.get_intervals('sigma_interval')
    # for i in range(sm_data.shape[0]):
    #     plt.subplot(sm_data.shape[0],1,i+1)
    #     plt.plot(smoother.smooth_data[i], linewidth=3, color='blue')
    #     plt.plot(smoother.data[i], '.k')
    #     plt.margins(0,0.2)
    #     plt.axhline(np.mean(smoother.smooth_data[i]), linestyle='dashed', color='red')
    #     plt.title(f"Mueller {i+1}"); plt.xlabel('time')
    #     plt.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.3)

    return smoother.smooth_data # 处理后的数据
    # return sm_data # 原始数据

def add_label(data, label):
    '''
    add_label:

    data:原始数据

    label: 标签
    
    return: 
    '''
    news_row = np.array([label] *data.shape[0])

    arr = np.vstack((data.T, news_row))
    
    return arr

def merge_data(data):
    '''
    data: 传入列表数据
    return: 合并后的数组 arr
    '''
    temp_data = data[0][:, 30000]

    for i in range(1, len(data)):
        ## 取前一万个数据
        merge_arr = np.concatenate((temp_data, data[i][:, 30000]), axis=1)
        temp_data = merge_arr

    return temp_data

def handle_data():
    '''
        return: 整个数据集 滤波后的数据
    '''
    all_data_test = []
    for i in range (0, 8):
        if i == 0:
            element_name = '../All_MM_MA_'
        else:
            element_name = '../All_MM_MA_0'
        path = element_name + str(i) + '.mat'
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

