#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
数据加载
"""

import gzip
import numpy as np
import pickle
from PIL import Image

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def load_single_cifar10(file_name):
    """
    加载后，将数据转换成 N × H × W × C
    :return: 训练和测试数据
    """
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    print(data.keys())
    # print(data['filenames'])
    raw_train_data = data['data']
    raw_train_data = np.reshape(raw_train_data, (10000, 3, 32, 32))
    train_data = np.transpose(raw_train_data, (0, 2, 3, 1))
    train_labels = data['labels']
    print(train_data.shape)
    print(train_data[0][0])
    print(data['data'].shape)
    print(len(data['labels']))
    print(data['data'][0])
    print(data['labels'][0])
    return train_data, train_labels


def load_raw_cifar10():
    """
    将图像缩放为指定的大小
    :return:
    """
    train_file_name = "../../data/cifar-10-batches-py/data_batch_1"
    train_data, train_labels = load_single_cifar10(train_file_name)
    all_train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    all_train_label = np.eye(10)[np.array(train_labels).reshape(-1)]
    for i in range(2, 6):
        print("i= ", i)
        train_file_name = "../../data/cifar-10-batches-py/data_batch_" + str(i)
        train_data, train_labels = load_single_cifar10(train_file_name)
        train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        new_label = np.eye(10)[np.array(train_labels).reshape(-1)]
        all_train_data = np.vstack((all_train_data, train_data))
        all_train_label = np.vstack((all_train_label, new_label))

    test_file_name = "../../data/cifar-10-batches-py/test_batch"
    test_data, test_labels = load_single_cifar10(test_file_name)
    test_data = (test_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    new_test_label = np.eye(10)[np.array(test_labels).reshape(-1)]

    return all_train_data, all_train_label, test_data, new_test_label
    # return test_data, test_labels


def load_diy_cifar(height, width):
    """
    将图像缩放为指定的大小, 并从数据中抽取部分数据，
    :return:
    """
    train_file_name = "../../data/cifar-10-batches-py/data_batch_1"
    train_data, train_labels = load_single_cifar10(train_file_name)
    data_resized = np.zeros((10000, height, width, 3))
    for i in range(10000):
        img = train_data[i]
        img = Image.fromarray(img)
        img = np.array(img.resize((height, width), Image.BICUBIC))
        data_resized[i, :, :, :] = img
    all_train_data = (data_resized - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    all_train_label = np.eye(10)[np.array(train_labels).reshape(-1)]
    for i in range(2, 6):
        print("i= ", i)
        train_file_name = "../../data/cifar-10-batches-py/data_batch_" + str(i)
        train_data, train_labels = load_single_cifar10(train_file_name)
        data_resized = np.zeros((10000, height, width, 3))
        for i in range(10000):
            img = train_data[i]
            img = Image.fromarray(img)
            img = np.array(img.resize((height, width), Image.BICUBIC))
            data_resized[i, :, :, :] = img
        data_resized = (data_resized - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        new_label = np.eye(10)[np.array(train_labels).reshape(-1)]
        all_train_data = np.vstack((all_train_data, data_resized))
        all_train_label = np.vstack((all_train_label, new_label))

    test_file_name = "../../data/cifar-10-batches-py/test_batch"
    test_data, test_labels = load_single_cifar10(test_file_name)
    test_data_resized = np.zeros((10000, height, width, 3))
    for i in range(10000):
        img = test_data[i]
        img = Image.fromarray(img)
        img = np.array(img.resize((height, width), Image.BICUBIC))
        test_data_resized[i, :, :, :] = img
    test_data_resized = (test_data_resized - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    test_label = np.eye(10)[np.array(test_labels).reshape(-1)]

    print("213--- ", train_data.shape)
    new_train_data, new_train_label = shufflelists([all_train_data, all_train_label])
    new_test_data, new_test_label = shufflelists([test_data_resized, test_label])
    return new_train_data[0:10000, :, :, :], new_train_label[0:10000, :], new_test_data[0:1000, :, :,:], new_test_label[0:1000, :]
    # return test_data, test_labels


def load_cifar10():
    with open("../../data/tmp", 'rb') as fo:
        data_load = pickle.load(fo, encoding='latin1')
    return data_load['data'], data_load['label'], data_load['test_data'], data_load['test_label']


def load_and_shuffle_cifar10():
    with open("../../data/tmp", 'rb') as fo:
        data_load = pickle.load(fo, encoding='latin1')
    new_data, new_label = shufflelists([data_load['data'], data_load['label']])
    new_test_data, new_test_label = shufflelists([data_load['test_data'], data_load['test_label']])
    return new_data, new_label, new_test_data, new_test_label


def shufflelists(lists):
    # 得到一个新的随机顺序,如[0,1,2,3,4]->[2,3,1,0,4]
    # 数据列表lists中的每个数据集的都会跟着这个随机顺序重新排列
    ri = np.random.permutation(len(lists[0]))
    out = []
    for l in lists:
        # 注意lists中的每个数据集l都是numpy.ndarray类型的数据
        # numpy.ndarray才可以以list、numpy.ndarray等序列类型作为下标,
        # 而list不能这样
        out.append(l[ri])
    return out


if __name__ == '__main__':
    data, label, test_data, test_label = load_diy_cifar(224, 224)
    print(data.shape)
    print(label[0:5])
    data_dic = {"data": data, "label": label, "test_data": test_data, "test_label": test_label}
    with open("../../data/tmp", 'wb') as fo:
        pickle.dump(data_dic, fo)

    # data, label, test_data, test_label = load_cifar10()
    # print(data.shape, "--", label.shape)
    # print(label[0:10])
    #
    # new_data, new_label = shufflelists([data, label])
    # print(new_data.shape, "--", new_label.shape)
    # print(new_label[0:10])

    # data2, label2, test_data2, test_label2 = load_mnist()
    # print(label2[0])
