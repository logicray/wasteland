#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
simple implement of bloom filter
"""

print(__doc__)


class BloomFilter:
    def __init__(self, size):
        self.size = size
        self.bit_array = [False] * size

    def hash1(self, n):
        return hash(n) % self.size

    def hash2(self, n):
        return hash(str(n)+str(n)) % self.size

    def push(self, n):
        tmp_index1 = self.hash1(n)
        self.bit_array[tmp_index1] = True

        tmp_index2 = self.hash2(n)
        self.bit_array[tmp_index2] = True

    def is_exist(self, n):
        bit_index1 = self.hash1(n)
        bit_index2 = self.hash2(n)
        if self.bit_array[bit_index1] and self.bit_array[bit_index2]:
            return True
        return False


def main():
    b_filter = BloomFilter(100)
    print(len(b_filter.bit_array))
    b_filter.push("a")
    b_filter.push("b")
    exist = b_filter.is_exist("b")
    print(exist)


if __name__ == '__main__':
    main()
