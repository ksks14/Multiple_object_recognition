"""
基于opencv的图像照度分类。
"""
import cv2 as cv
import numpy as np


def class_fi_cation(image, new_data=None):
    data = cv.imread(image)
    data = data.astype(np.uint8)
    print(type(data))
    print(data.dtype)
    print(data.shape)
    cv.convertScaleAbs(data, new_data, 1.5, 10)

    cv.imshow('input', data)
    cv.imshow('input_new', new_data)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    data_path = '../src/data_test/test.png'
    class_fi_cation(data_path)
