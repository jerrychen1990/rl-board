import multiprocessing
import time
import types
from abc import ABC
from multiprocessing.managers import BaseManager

#
# class C:
#     def __init__(self, n):
#         self.n = n
#
#     def add_n(self):
#         self.n +=1
#
#     def get_n(self):
#         return self.n
#
#
# def produce(conn, c):
#     for idx in range(10):
#         #         print(f"send:{idx}")
#         #         conn.send(idx)
#         c.add_n()
#         # print(c.n)
#         time.sleep(1)
#
#
# def custom(conn, c):
#     for dix in range(10):
#         print(f"c.n:{c.get_n()}")
#         time.sleep(2)
#
#
# BaseManager.register('C', C, exposed=['add_n', 'get_n'])
#
#
from typing import Iterable

from snippets import set_kwargs


class A:
    def __init__(self):
        self.n = 0


def add(a):
    a.n += 1


from multiprocessing import Process


class BaseProcess(Process):
    def __init__(self, a, *args, **kwargs):
        super(BaseProcess, self).__init__(*args, **kwargs)
        self.b = a

    def run(self):
        # add(self.b)
        self.b.n = 10
        # print(self.b.n)


def set_args(func):
    def wrapped(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        func(self, **kwargs)

    return wrapped


class A:
    @set_args
    def __init__(self, a, b):
        print("done")


class B(A):
    pass


class C(ABC):
    num: str = NotImplemented


class D(C):
    pass


class A:
    a = 1
    b = 2


def get_cls(b):
    rs = types.new_class(name="B", bases=(A,), kwds={}, exec_body=lambda x: x.update(b=b))
    return rs


def tuplize(v):
    if isinstance(v, list) or isinstance(v, tuple):
        return tuple([tuplize(e) for e in v])
    return v


tmp = ([['_', '_', '_', '_'],
        ['_', '_', '_', '_'],
        ['O', 'X', 'O', '_'],
        ['X', 'X', 'O', '_']],
       'X')


class A(object):
    @set_kwargs(ignores=["b"])
    def __init__(self, a, b):
        self.c = a + b


if __name__ == "__main__":
    # print(tuplize(tmp))

    tmp = A(1, b=2)
    print(tmp.__dict__)

    # B = get_cls(4)
    # print(B.a)
    # print(B.b)

    # a = B(a=2, b=3)
    # print(a.a)
    # print(a.b)

    # process = BaseProcess(a=a)
    # process.start()
    # process.join()
    # print(process.b.n)
    # print(a.n)
