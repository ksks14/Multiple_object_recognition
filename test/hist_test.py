import cv2 as cv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from utils.times import func_time
"""
这里原则上能用numpy就用numpy，因为numpy是利用c完成的，所以它可以做到尽可能的挽救python带来的性能劣势，虽然不及c++，但是能在python本身的速度上达到处理的极致。
"""

# @func_time
def hist_num_by_np(img_path):
    """

    :param img_path:
    :return:
    """
    # Read a img data
    img = cv.imread(img_path)
    # Copy a grayscale map for classification algorithm
    img_copy = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # make a hist matrix
    img_copy = img_copy.flatten()
    num_matrix = np.bincount(img_copy)
    # 将统计到的数据作为特征向量返回。
    return num_matrix

@func_time
def hist_num_by_Counter(img_path):
    """

    :param img_path:
    :return:
    """
    img = cv.imread(img_path)
    # Copy a grayscale map for classification algorithm
    img_copy = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_copy = img_copy.flatten()
    num_matrix = Counter(img_copy)
    return num_matrix



if __name__ == '__main__':
    img_path = '../src/data_test/test_5.png'
    img = cv.imread(img_path)
    print(img.shape)
    hist_num_by_np(img_path)

