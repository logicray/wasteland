#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
input a number n
output list of all primers less than n
"""
import time
import math
import numpy as np
from collections import deque


def is_primer(x):
    sqrt_x = int(math.sqrt(x))+1
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


def sieve_erato(n):
    """
    埃氏筛法(Sieve of Eratosthenes, SoE)
    :param n:
    :return:
    """
    p = [True] * (n + 1)
    p[0] = p[1] = False
    for i in range(2, int(math.sqrt(n) + 1)):
        if p[i]:
            for j in range(i * i, n + 1, i):
                p[j] = False
    prime = [i for i in range(n + 1) if p[i]]
    print(len(prime))
    return prime


def sieve_euler(n):
    """
    欧拉筛法(Sieve of Euler)
    :param n:
    :return:
    """
    prime_list = deque()
    p = [True] * (n + 1)  # true 表示是素数
    p[0] = p[1] = False
    for i in range(2, n):
        if p[i]:
            # prime_list.append(i)
            prime_list.append(i)
        for j in prime_list:
            t = i * j
            if t > n:
                break
            p[t] = False
            # 每个合数只被他的最小质因数筛去
            if i % j == 0:
                break

    # prime = [i for i in range(n + 1) if p[i]]
    print(len(prime_list))
    # print(prime)
    # print(prime_list)
    return prime_list


def check_equal(n):
    traversal_res = traversal(n)
    sieve_erato_res = sieve_euler(n)
    # print(traversal_res, sieve_erato_res)
    if traversal_res == sieve_erato_res:
        return True
    return False


if __name__ == '__main__':
    start = time.time()
    res = sieve_euler(2000000)
    print(len(res))
    # print(res[-20:])
    print("time consume ", time.time() - start)
    # print(check_equal(40))
