#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
decorator exercise
"""
from functools import wraps


def hi(name="gross"):
    print("in hi func")

    def welcome():
        return "in welcome"

    def greet():
        return "in greet func"

    if name == "gross":
        return welcome
    else:
        return greet


def before_hi(func):
    print("do something before hi")
    func()


greet2 = hi()

# print(hi())
print(greet2)
print(greet2())

before_hi(hi)


def simple_decorator(func):
    @wraps(func)
    def wrap_func():
        print("something before func")
        print(func())
        print("something after func")

    return wrap_func


def func_need_decorate():
    return "need decorate"


@simple_decorator
def func_need_decorate2():
    return "need decorate2"


print(func_need_decorate())
x = simple_decorator(func_need_decorate)

x()
print(x.__name__)
print("split line -----")
func_need_decorate2()
print(func_need_decorate2.__name__)


def logit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        print(f.__name__ + " was called")
        return f(*args, **kwargs)

    return decorated


def logit2(log_file="a1.log"):
    def log_decorate(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            print(f.__name__ + " was called")
            print("here to create log file:", log_file)
            return f(*args, **kwargs)

        return decorated
    return log_decorate

@logit2()
def add(a, b):
    return a + b


print(add(3, 5))


class LogitClass(object):
    def __init__(self, log_file="out.log"):
        self.log_file = log_file

    def __call__(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            print(f.__name__ + " was called")
            print("here to create log file:", self.log_file)
            return f(*args, **kwargs)
        return decorated

    def notify(self):
        pass


@LogitClass()
def add2(a, b):
    return a + b


print(add2(3, 6))


