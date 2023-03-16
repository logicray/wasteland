#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
cifar识别，convolution network
"""

import tensorflow as tf
from loader import load_data
import math

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def lr_compute(step):
    max_lr, min_lr, decay_speed = 0.003, 0.0001, 2000.0
    return min_lr + (max_lr - min_lr) * math.exp(-step / decay_speed)


def main(argv=None):
    train_data, train_label, test_data, test_label = load_data.load_raw_cifar10()
    print(train_label.shape)
    x_in = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    # t is label
    t = tf.placeholder(tf.float32, shape=(None, 10))
    K = 16
    L = 48
    M = 96
    N = 200

    # y = tf.nn.softmax(tf.matmul(x, w) + b)
    pkeep = tf.placeholder(tf.float32)

    #
    w1 = tf.Variable(tf.truncated_normal([6, 6, 3, K], stddev=0.1))
    b1 = tf.Variable(tf.ones([K]) / 10)
    y1 = tf.nn.relu(tf.nn.conv2d(x_in, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    y1d = tf.nn.dropout(y1, pkeep)
    y1_pool = tf.nn.max_pool(y1d, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #
    w2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    b2 = tf.Variable(tf.ones([L]) / 10)
    y2 = tf.nn.relu(tf.nn.conv2d(y1_pool, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
    y2d = tf.nn.dropout(y2, pkeep)

    w3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    b3 = tf.Variable(tf.ones([M]) / 10)
    y3 = tf.nn.relu(tf.nn.conv2d(y2d, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
    y3d = tf.nn.dropout(y3, pkeep)

    w4 = tf.Variable(tf.truncated_normal([8 * 8 * M, N], stddev=0.1))
    b4 = tf.Variable(tf.ones([N]) / 10)
    yy = tf.reshape(y3d, shape=[-1, 8 * 8 * M])
    y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
    y4d = tf.nn.dropout(y4, pkeep)

    w5 = tf.Variable(tf.truncated_normal([N, N], stddev=0.1))
    b5 = tf.Variable(tf.ones([N]) / 10)
    y5 = tf.nn.relu(tf.matmul(y4d, w5) + b5)
    y5d = tf.nn.dropout(y5, pkeep)

    w6 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    b6 = tf.Variable(tf.ones([10]) / 10)
    logits = tf.matmul(y5d, w6) + b6
    y = tf.nn.softmax(logits)
    print("y5_pool.shape", logits.get_shape().as_list())
    c_e_list = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=t)
    print("y5_pool.shape", c_e_list.get_shape().as_list())
    cross_entropy = tf.reduce_mean(c_e_list)
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    lr = tf.placeholder(tf.float32, shape=[])
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    batch_size = 100
    init_op = tf.global_variables_initializer()
    # print(train_data.shape)
    # print(train_label.shape)
    with tf.Session(config= tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options={'allow_growth': True})) as sess:
        # tf.initialize_all_variables().run()
        sess.run(init_op)
        for step in range(4000):
            start = (step * batch_size) % 10000
            end = start + batch_size
            batch_xs = train_data[start:end, :, :, :]
            batch_ys = train_label[start:end, :]
            # print('==', x.get_shape())
            # print('==', w.get_shape())
            # print('==', batch_xs.shape)
            sess.run(train_step, feed_dict={x_in: batch_xs, t: batch_ys, lr: lr_compute(step), pkeep: 0.75})
            # print(type(x))
            if step % 100 == 0:
                # train set metric
                acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x_in: batch_xs, t: batch_ys, pkeep: 1})
                # print('train acc,loss:',acc,loss)
                test_acc, test_loss = sess.run([accuracy, cross_entropy],
                                               feed_dict={x_in: test_data, t: test_label, pkeep: 1})
                # t2 = time.time()
                # end_time = int(round(t2 * 1000))
                print('train acc,loss:', acc, loss, test_acc, test_loss)


if __name__ == '__main__':
    tf.app.run()
