#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
mnist识别，convolution network
"""

import tensorflow as tf
from loader import load_data
import math
import os
import time

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# cfg = tf.ConfigProto(gpu_options={'allow_growth': True})
# K.set_session(tf.Session(config=cfg))

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
# config.gpu_options.polling_inactive_delay_msecs = 10
# session = tf.compat.v1.Session(config=config)


def lr_compute(step):
    max_lr, min_lr, decay_speed = 0.003, 0.0001, 2000.0
    return min_lr + (max_lr - min_lr) * math.exp(-step / decay_speed)


def main():
    train_data, train_label, test_data, test_label = load_data.load_mnist()

    # x_in = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    # t is label
    # t = tf.placeholder(tf.float32, shape=(None, 10))

    K = 4
    L = 8
    M = 12
    N = 200

    # init_op = tf.global_variables_initializer()
    # init_op = tf.initialize_all_variables()
    # y = tf.nn.softmax(tf.matmul(x, w) + b)
    pkeep = tf.placeholder(tf.float32)

    # x_in is N , 28 , 28 , 1
    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))
    b1 = tf.Variable(tf.ones([K]) / 10)
    y1 = tf.nn.relu(tf.nn.conv2d(x_in, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    # y1d = tf.nn.dropout(y1, pkeep)

    # y1  shape is N , 28 , 28 , 4
    w2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    b2 = tf.Variable(tf.ones([L]) / 10)
    y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
    # y2d = tf.nn.dropout(y2, pkeep)

    # y2 shape is N , 14 , 14 , 8
    w3 = tf.Variable(tf.truncated_normal([7, 7, L, M], stddev=0.1))
    b3 = tf.Variable(tf.ones([M]) / 10)
    y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)
    # y3d = tf.nn.dropout(y3, pkeep)

    # y3 shape is N ,7 , 7 , 12
    w4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    b4 = tf.Variable(tf.ones([N]) / 10)
    # yy shape is N, 7*7*12
    yy = tf.reshape(y3, shape=[-1, 7 * 7 * M])
    y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
    # y4d = tf.nn.dropout(y4, pkeep)

    # y4 shape is N , N
    w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    b5 = tf.Variable(tf.ones([10]) / 10)
    logits = tf.matmul(y4, w5) + b5
    y = tf.nn.softmax(logits)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=t))
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    lr = tf.placeholder(tf.float32, shape=[])
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    batch_size = 100
    init_op = tf.initialize_all_variables()
    # print(train_data.shape)
    # print(train_label.shape)

    # tf.initialize_all_variables().run()
    for step in range(4000):
        if step % 100 == 0:
            t_now = time.time()
            start_time = int(round(t_now * 1000))
        start = (step * batch_size) % 60000
        end = start + batch_size
        batch_xs = train_data[start:end, :, :, :]
        batch_ys = train_label[start:end, :]
        # print('==', x.get_shape())
        # print('==', w.get_shape())
        # print('==', batch_xs.shape)
        sess.run(train_step, feed_dict={x_in: batch_xs, t: batch_ys, lr: lr_compute(step), pkeep: 0.75})
        # print(type(x))
        if step % 100 == 0:
            # pass
            # train set metric
            acc, loss = sess.run(fetches=[accuracy, cross_entropy],
                                 feed_dict={x_in: batch_xs, t: batch_ys, pkeep: 1})
            # print('train acc,loss:',acc,loss)
            test_acc, test_loss = sess.run([accuracy, cross_entropy],
                                           feed_dict={x_in: test_data, t: test_label, pkeep: 1})
            t2 = time.time()
            end_time = int(round(t2 * 1000))
            print('train acc,loss:', acc, loss, test_acc, test_loss, "time in ms: ", end_time - start_time)
