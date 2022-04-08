#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     test.py
   Author :       chenhao
   time：          2022/4/6 16:48
   Description :
-------------------------------------------------
"""


def f(d):
    rs = 1
    for i in range(1, d + 1):
        rs *= i
    return rs


for i in range(10):
    print(i)
    print(f(i))
