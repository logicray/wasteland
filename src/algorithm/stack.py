#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
a simple implement of stack
can not automatic expansion
"""


class Stack:
    def __init__(self, size):
        self.size = size
        self.stack = []
        self.top = -1

    def push(self, x):
        if self.is_full():
            raise Exception("stack is full")
        else:
            self.stack.append(x)
            self.top += 1

    def pop(self):
        if self.is_empty():
            raise Exception("stack is empty")
        else:
            self.top -= 1
            self.stack.pop()

    def is_full(self):
        return self.top + 1 == self.size

    def is_empty(self):
        return self.top == -1

    def show(self):
        print(self.stack)


def main():
    try:
        stack = Stack(10)
        stack.push(1)
        stack.push(2)
        stack.push(3)
        stack.show()
        stack.pop()
        stack.show()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
