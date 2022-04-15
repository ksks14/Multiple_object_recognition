import os
import cv2 as cv
import numpy as np
from utils.hist_test import hist_num_by_np
from utils.times import func_time

def get_paths(root):
    """

    :param root:
    :return:
    """
    files = os.listdir(root)
    # 先修改图片尺寸，使得他们一样大
    # for name in files:
    #     data_path = os.path.join(root,name)
    #     data = cv.imread(data_path)
    #     data = cv.resize(data,(500,500))
    #     cv.imwrite(data_path,data)
    return files


def make_label_files(root, label_path='labels.txt'):
    """

    :param root:
    :return:
    """
    file_names = get_paths(root)
    with open(os.path.join(root, label_path), 'w', encoding='utf-8') as f:
        for names in file_names:
            if not names == label_path:
                f.write(names + ":\n")


def get_data(root):
    """

    :param root:
    :return:
    """
    files = get_paths(root)
    data_list = []
    for file in files:
        if file.split('.')[1] == 'png':
            print(os.path.join(root,file))
            data = hist_num_by_np(os.path.join(root,file))
            data_list.append(data)
    data = np.array(data_list,dtype=object)
    return data


def get_labels(root, label_path='labels.txt'):
    """
    获取实值
    :param root:
    :param label_path:
    :return:
    """
    labels = []
    with open(os.path.join(root, label_path)) as file_read:
        labels = np.array([int(line[-2]) for line in file_read.readlines()])
        return labels

@func_time
def get_datas_labels(root):
    datas = get_data(root)
    labels = get_labels(root)
    return datas,labels


if __name__ == '__main__':
    root_path = '../src/data_test'
    data,labels = get_datas_labels(root_path)
