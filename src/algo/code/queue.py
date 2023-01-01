#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
simple implement of queue
FIFO
"""


class Queue:
    def __init__(self, size):
        self.size = size
        self.queue = []
        self.f = 0
        self.r = 0

    def add(self, x):
        if self.is_full():
            raise Exception("queue is full")
        self.r += 1
        self.queue.append(x)

    def delete(self):
        if self.is_empty():
            raise Exception("queue is empty")
        self.f += 1
        self.queue.remove(self.f)

    def is_empty(self):
        if self.f == self.r:
            return True
        return False

    def is_full(self):
        if self.f == self.r + self.size:
            return True
        return False

    def show(self):
        print(self.queue)


if __name__ == '__main__':
    queue = Queue(10)
    queue.add(1)
    queue.add(2)
    queue.add(3)
    queue.delete()
    queue.delete()
    queue.show()



