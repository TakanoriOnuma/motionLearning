#!/usr/bin/env python
# coding:utf-8

# キリのいい数値か調べる
def isRoundNumber(num):
    if num == 0:
        return True
    digits = []
    while num > 0:
        digits.append(num % 10)
        num /= 10
    digits = digits[:-1]
    return all(digit == 0 for digit in digits)

# 丁度対数スケールになっているか調べる
def isJustLogScale(num):
    import math
    if num < 10:
        return False
    log = math.log10(num)
    return (log - math.ceil(log)) == 0.0

# 小数のループをyieldで生成する
def drange(begin, end, step):
    n = begin
    while n <= end + step / 10:
        yield n
        n += step
