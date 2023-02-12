#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
tf.feature_column 实践
"""

import tensorflow as tf


def test_feature_column():
    features = {
        "numbers": [[1],
                    [2],
                    [3],
                    [-1],
                    [0],
                    [5],
                    [-2],
                    [-3],
                    [-4]
                    ]
    }

    numbers = tf.feature_column.categorical_column_with_identity("numbers", num_buckets=4, default_value=1)

    # numbers = tf.feature_column.indicator_column(numbers)
    numbers = tf.feature_column.embedding_column(numbers, 4)
    inputs = tf.feature_column.input_layer(features, [numbers])
    print(numbers)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.tables_initializer())
        v = sess.run(inputs)
        print(v)


def test_embedding():
    color_data = {
        "color": [["R"], ["G"], ["B"], ["R"]]
    }
    color_column = tf.feature_column.categorical_column_with_vocabulary_list("color", ["R", "G", "B"])
    # color_column = tf.feature_column.indicator_column()
    color_embedding = tf.feature_column.embedding_column(color_column, 3)
    inputs = tf.feature_column.input_layer(color_data, [color_embedding])
    # builder =  _LazyBuilder(color_data)
    color_column_tensor = color_column._get_sparse_tensors(color_data)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.tables_initializer())
        v = sess.run(inputs)
        print(v)
        print(color_column_tensor.id_tensor)


if __name__ == '__main__':
    test_embedding()
