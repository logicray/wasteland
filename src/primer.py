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


def sieve_era(n):
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
    p = [True] * (n + 1)
    p[0] = p[1] = False
    for i in range(2, int(math.sqrt(n) + 1)):
        if p[i]:
            for j in range(i * i, n + 1, i):
                p[j] = False
    prime = [i for i in range(n + 1) if p[i]]
    print(len(prime))




    return prime


def sieve_linear(n):
    """
    欧拉筛法(Sieve of Euler)
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


if __name__ == '__main__':
    start = time.time()
    res = traversal(2000000)
    print(len(res))
    print(res[-20:])
    print("time consume ", time.time() - start)
