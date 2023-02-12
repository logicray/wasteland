#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
加载text8 文件，并解析为单词列表（未去重）
"""

import zipfile
import csv
import os
import collections
from loader import BASE_DATA_DIR

Dataset = collections.namedtuple('Dataset', ['features', 'label'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_data(filename):
    f = zipfile.ZipFile(filename)
    for n in f.namelist():
        return f.read(n).split()
    f.close()


def _load_csv(file_name) -> list:
    """

    :param file_name:
    :return:
    """
    res = []
    full_path_file = os.path.join(BASE_DATA_DIR, file_name)
    with open(full_path_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            res.append(row)
    return res


def load_dbpedia():
    """

    :param file_name:
    :return:
    """
    train_data = _load_csv("dbpedia_csv/train.csv")
    train_features = []
    train_label = []

    for line in train_data:
        train_label.append(line[0])
        train_features.append(line[1:])

    train_dataset = Dataset(train_features, train_label)

    test_data = _load_csv("dbpedia_csv/test.csv")
    test_features = []
    test_label = []

    for line in test_data:
        test_label.append(line[0])
        test_features.append(line[1:])
    test_dataset = Dataset(test_features, test_label)

    return Datasets(train_dataset, None, test_dataset)


if __name__ == '__main__':
    content = load_data("../../data/text8.zip")
    print(len(content) / 1000 / 1000)
    print(content[0:10])
    x = load_dbpedia()
    print(x[0])
