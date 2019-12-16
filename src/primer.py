#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
input a number n
output list of all primers less than n
"""
import time
import math


def is_primer(x):
    sqrt_x = int(math.sqrt(x))
    for i in range(2, sqrt_x):
        if x % i == 0:
            return False
    return True


def traversal(n):
    p_list = [2]
    """
    遍历法
    :param n:
    :return:
    """
    for x in range(3, n):
        if is_primer(x):
            # print(x)
            p_list.append(x)
    return p_list


if __name__ == '__main__':
    start = time.time()
    res = traversal(1000000)
    print(len(res))
    print(res[-20:])
    print("time consume ", time.time() - start)
