#!/usr/bin/env python3
# -*- coding:utf8 -*-

import tensorflow as tf

def tt():
    a = tf.placeholder(dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)

    c = tf.add(a, b)
    d = tf.sin(a)

    e = tf.math.multiply(c, d)

    f = tf.cos(c)

    with tf.Session() as sess:
        res = sess.run(f, feed_dict={a: 2, b: 3})
        print(res)

        x = [
            [1., 1.],
            [2., 2.]
        ]
        print(sess.run(tf.reduce_mean(x)))
        print(sess.run(tf.reduce_mean(x, 0)))
        print(sess.run(tf.reduce_mean(x, 1)))


def func(param1, c='', d=''):
    print(param1)
    print(c)


def func1(param1, *param2):
    print(param1)
    print(param2)


def func2(param1, **param2):
    print(param1)
    print(param2)


if __name__ == '__main__':
    print('.', end='-')
    print()
    x = ['aa', 'bb']
    func(*x)
    y = {'c': 'cc', 'd': 'dd'}
    # print(type(**y))
    func('a', **y)
    func1('xx', 'yy', 'zz')
    func2('xx', y='y', z='z')



