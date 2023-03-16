#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
mnist识别，多种模型
"""

import tensorflow as tf
import math
from loader import load_data

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def lr_compute(step):
    """
    learning rate compute
    :param step:
    :return:
    """
    max_lr, min_lr, decay_speed = 0.003, 0.0001, 2000.0
    return min_lr + (max_lr - min_lr) * math.exp(-step / decay_speed)


def single_layer_perceptron(features, labels, lr, p_keep=None):
    x = tf.reshape(features, [-1, 784])
    # print('--',x.get_shape().as_list())
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # init_op = tf.global_variables_initializer()

    y = tf.nn.softmax(tf.matmul(x, w) + b)
    cross_entropy = -tf.reduce_mean(labels * tf.log(y))
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    train_step = optimizer.minimize(cross_entropy)
    return accuracy, cross_entropy, train_step


def multi_layer_perceptron(features, labels, lr, p_keep=None):
    x = tf.reshape(features, [-1, 784])
    # print('--',x.get_shape().as_list())
    # w = tf.Variable(tf.zeros([784,10]))
    # b = tf.Variable(tf.zeros([10]))
    K = 200
    L = 100
    M = 60
    N = 30
    w1 = tf.Variable(tf.truncated_normal([28 * 28, K], mean=0.0, stddev=0.1))
    b1 = tf.Variable(tf.zeros([K]))

    w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
    b2 = tf.Variable(tf.zeros([L]))

    w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    b3 = tf.Variable(tf.zeros([M]))

    w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    b4 = tf.Variable(tf.zeros([N]))

    w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    b5 = tf.Variable(tf.zeros([10]))
    # y = tf.nn.softmax(tf.matmul(x, w) + b)
    y1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
    y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3)
    y4 = tf.nn.sigmoid(tf.matmul(y3, w4) + b4)
    y = tf.nn.softmax(tf.matmul(y4, w5) + b5)

    cross_entropy = -tf.reduce_sum(labels * tf.log(y))
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    train_step = optimizer.minimize(cross_entropy)
    return accuracy, cross_entropy, train_step


def mlp_with_dropout(features, labels, lr, p_keep):
    x = tf.reshape(features, [-1, 784])
    # print('--',x.get_shape().as_list())
    K = 200
    L = 100
    M = 60
    N = 30
    w1 = tf.Variable(tf.truncated_normal([28 * 28, K], stddev=0.1))
    b1 = tf.Variable(tf.ones([K]) / 10)

    w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
    b2 = tf.Variable(tf.ones([L]) / 10)

    w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    b3 = tf.Variable(tf.ones([M]) / 10)

    w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    b4 = tf.Variable(tf.ones([N]) / 10)

    w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    b5 = tf.Variable(tf.ones([10]) / 10)

    # init_op = tf.global_variables_initializer()
    # init_op = tf.initialize_all_variables()
    # y = tf.nn.softmax(tf.matmul(x, w) + b)
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    y1d = tf.nn.dropout(y1, p_keep)

    y2 = tf.nn.relu(tf.matmul(y1d, w2) + b2)
    y2d = tf.nn.dropout(y2, p_keep)

    y3 = tf.nn.relu(tf.matmul(y2d, w3) + b3)
    y3d = tf.nn.dropout(y3, p_keep)

    y4 = tf.nn.relu(tf.matmul(y3d, w4) + b4)
    y4d = tf.nn.dropout(y4, p_keep)

    logits = tf.matmul(y4d, w5) + b5
    y = tf.nn.softmax(logits)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    return accuracy, cross_entropy, train_step


def mlp_with_relu(features, labels, lr, p_keep):
    # t is label
    x = tf.reshape(features, [-1, 784])
    # print('--',x.get_shape().as_list())
    # w = tf.Variable(tf.zeros([784,10]))
    # b = tf.Variable(tf.zeros([10]))
    K = 200
    L = 100
    M = 60
    N = 30
    w1 = tf.Variable(tf.truncated_normal([28 * 28, K], stddev=0.1))
    b1 = tf.Variable(tf.ones([K]) / 10)

    w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
    b2 = tf.Variable(tf.ones([L]) / 10)

    w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    b3 = tf.Variable(tf.ones([M]) / 10)

    w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    b4 = tf.Variable(tf.ones([N]) / 10)

    w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    b5 = tf.Variable(tf.ones([10]) / 10)

    # init_op = tf.global_variables_initializer()
    # init_op = tf.initialize_all_variables()
    # y = tf.nn.softmax(tf.matmul(x, w) + b)

    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
    y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)
    y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)
    logits = tf.matmul(y4, w5) + b5
    y = tf.nn.softmax(logits)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    return accuracy, cross_entropy, train_step


def main(argv=None):
    train_data, train_label, test_data, test_label = load_data.load_mnist()

    # x_in is features, t is labels
    x_in = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    t = tf.placeholder(tf.float32, shape=(None, 10))
    lr = tf.placeholder(tf.float32, shape=[])
    p_keep = tf.placeholder(tf.float32)
    accuracy, cross_entropy, train_step = single_layer_perceptron(x_in, t, lr, p_keep)

    batch_size = 500
    init_op = tf.initialize_all_variables()
    # print(train_data.shape)
    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(4000):
            start = (step * batch_size) % 60000
            end = start + batch_size
            batch_xs = train_data[start:end, :, :, :]
            batch_ys = train_label[start:end, :]
            # print('==', x.get_shape())
            # print('==', w.get_shape())
            # print('==', batch_xs.shape)
            sess.run(train_step, feed_dict={x_in: batch_xs, t: batch_ys, lr: lr_compute(step), p_keep: 0.75})
            # print(type(x))
            if step % 100 == 0:
                # train set metric
                acc, loss = sess.run([accuracy, cross_entropy],
                                     feed_dict={x_in: batch_xs, t: batch_ys, p_keep: 0.75})
                # print('train acc,loss:',acc,loss)
                test_acc, test_loss = sess.run([accuracy, cross_entropy],
                                               feed_dict={x_in: test_data, t: test_label, p_keep: 0.75})
                s = "train acc:{0:.2f},loss:{1:.2f},---test acc:{2:.2f},loss:{3:.2f}".format(acc, loss, test_acc,
                                                                                             test_loss)
                print(s)


if __name__ == '__main__':
    tf.app.run()
