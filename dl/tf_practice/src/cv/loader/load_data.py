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
BASE_DATA_DIR = "/home/page/projects/deeplab/data/"


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    # Convert to dense 1-hot representation.
    return (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)


def load_mnist():
    """
    mnist 数据加载，训练样本数为60000, 测试样本数为10000,图像大小为 28×28,单通道
    :return:
    """
    train_data = extract_data("../../data/train-images-idx3-ubyte.gz", 60000)
    train_label = extract_labels('../../data/train-labels-idx1-ubyte.gz', 60000)
    test_data = extract_data('../../data/t10k-images-idx3-ubyte.gz', 10000)
    test_labels = extract_labels('../../data/t10k-labels-idx1-ubyte.gz', 10000)
    return train_data, train_label, test_data, test_labels


def load_batch_cifar10(file_name):
    """
    :return: 加载的训练数据和标签
    """
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    print(data.keys())
    # print(data['filenames'])
    raw_train_data = data['data']
    raw_train_label = data['labels']
    print(raw_train_data.shape)
    print(len(data['labels']))
    return raw_train_data, np.array(raw_train_label)


def merge_cifar10_train():
    """
    加载后，将数据转换成 N × H × W × C
    :return: 5合1训练和数据
    """
    train_file_name = BASE_DATA_DIR + "cifar-10-batches-py/data_batch_1"
    train_data, train_labels = load_batch_cifar10(train_file_name)
    print(train_data[0][0])
    print("train label example: ", train_labels[0:3])
    for i in range(2, 6):
        print("i= ", i)
        train_file_name = BASE_DATA_DIR + "cifar-10-batches-py/data_batch_" + str(i)
        train_data_i, train_labels_i = load_batch_cifar10(train_file_name)
        train_data = np.vstack((train_data, train_data_i))
        train_labels = np.hstack((train_labels, train_labels_i))

    print("train data shape: ", train_data.shape)
    print("train data example: ", train_data[0][0])
    print("train label shape:", train_labels.shape)
    print("train label example: ", train_labels[0:3])

    data_dic = {"data": train_data, "label": train_labels}
    with open(BASE_DATA_DIR + "cifar-10-batches-py/data_batch_all", 'wb') as fo:
        pickle.dump(data_dic, fo)
    return train_data, train_labels


def load_single_cifar10(file_name):
    """
    加载后，将样本数据转换成 N × H × W × C
    label数据进行one-hot编码
    :return: 转换后的数据
    """
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    print(data.keys())
    # print(data['filenames'])
    raw_train_data = data['data']
    raw_train_data = np.reshape(raw_train_data, (10000, 3, 32, 32))
    train_data = np.transpose(raw_train_data, (0, 2, 3, 1))
    train_labels = data['labels']
    # new_label = np.eye(10)[np.array(train_labels).reshape(-1)]
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
    train_file_name = BASE_DATA_DIR + "cifar-10-batches-py/data_batch_1"
    train_data, train_labels = load_single_cifar10(train_file_name)
    all_train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    all_train_label = np.eye(10)[np.array(train_labels).reshape(-1)]
    for i in range(2, 6):
        print("i= ", i)
        train_file_name = BASE_DATA_DIR + "cifar-10-batches-py/data_batch_" + str(i)
        train_data, train_labels = load_single_cifar10(train_file_name)
        train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        new_label = np.eye(10)[np.array(train_labels).reshape(-1)]
        all_train_data = np.vstack((all_train_data, train_data))
        all_train_label = np.vstack((all_train_label, new_label))

    test_file_name =  BASE_DATA_DIR + "cifar-10-batches-py/test_batch"
    test_data, test_labels = load_single_cifar10(test_file_name)
    test_data = (test_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    new_test_label = np.eye(10)[np.array(test_labels).reshape(-1)]

    return all_train_data, all_train_label, test_data, new_test_label
    # return test_data, test_labels


def trans_data(raw_data, height, width):
    """

    :param raw_data:
    :return:
    """
    size = len(raw_data)
    raw_train_data = np.reshape(raw_data, (size, 3, 32, 32))
    train_data = np.transpose(raw_train_data, (0, 2, 3, 1))

    data_resized = np.zeros((size, height, width, 3))
    for i in range(size):
        img = train_data[i]
        img = Image.fromarray(img)
        img = np.array(img.resize((height, width), Image.BICUBIC))
        data_resized[i, :, :, :] = img
    all_train_data = (data_resized - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    return all_train_data


def choice_cifar10(height, width, train_n, test_n):
    """
    将图像缩放为指定的大小, 并从数据中抽取部分数据，然后保存
    :return:
    """
    train_file_name = "../../data/cifar-10-batches-py/data_batch_all"
    with open(train_file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        train_data = data['data']
        train_labels = data['label']
    # 读取数据后，先抽样
    new_train_data, new_train_label = shufflelists([train_data, train_labels])
    choice_train_data = new_train_data[0:train_n, :]
    choice_train_label = new_train_label[0:train_n]
    all_train_label = np.eye(10)[np.array(choice_train_label).reshape(-1)]
    all_train_data = trans_data(choice_train_data, height, width)


    test_file_name = "../../data/cifar-10-batches-py/test_batch"
    with open(test_file_name, 'rb') as fo:
        test_load_data = pickle.load(fo, encoding='latin1')
        test_data = test_load_data['data']
        test_labels = np.array(test_load_data['labels'])
    new_test_data, new_test_label = shufflelists([test_data, test_labels])
    choice_test_data = new_test_data[0:test_n, :]
    choice_test_label = new_test_label[0:test_n]

    all_test_data = trans_data(choice_test_data, height, width)
    all_test_label = np.eye(10)[np.array(choice_test_label).reshape(-1)]
    return all_train_data, all_train_label, all_test_data, all_test_label


def load_cifar10():
    with open("../../data/tmp", 'rb') as fo:
        data_load = pickle.load(fo, encoding='latin1')
        all_train_label = np.eye(10)[np.array(choice_train_label).reshape(-1)]
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
    # load_diy_cifar(224, 224)
    # print(data.shape)
    # print(label[0:5])
    # data_dic = {"data": data, "label": label, "test_data": test_data, "test_label": test_label}
    # with open("../../data/tmp", 'wb') as fo:
    #     pickle.dump(data_dic, fo)

    # data, label, test_data, test_label = load_cifar10()
    # print(data.shape, "--", label.shape)
    # print(label[0:10])
    #
    # new_data, new_label = shufflelists([data, label])
    # print(new_data.shape, "--", new_label.shape)
    # print(new_label[0:10])

    # data2, label2, test_data2, test_label2 = load_mnist()
    # print(label2[0])
    train_data, train_label, test_data, test_label = choice_cifar10(224, 224, 1000, 100)
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    # merge_cifar10_train()
