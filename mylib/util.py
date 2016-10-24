#!/usr/bin/env python
# coding:utf-8

import os, subprocess

# フォルダの作成
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

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

# 対数スケールのジェネレータ
# ※1, 10, 100などの10の対数スケールの値を渡すこと
# ※beginが0の時は例外的に0を出したあと1から対数スケールで進める
def logrange(begin, end):
    if begin == 0:
        yield 0
        begin = 1
    n    = begin
    step = begin
    yield n
    while n < end:
        for cnt in range(9):
            n += step
            yield n
        step *= 10

# gnuplotを実行する
def doGnuplot(arg, exeName):
    p = subprocess.Popen(['gnuplot', '-e', arg, exeName], stderr=subprocess.PIPE)
    # エラーがあったら出力する
    for line in p.stderr:
        print line,

# pythonを実行する
def doPython(exeName, *args):
    cmds = ['python']
    cmds.append(exeName)
    cmds.extend(args)
    p = subprocess.Popen(cmds, stderr=subprocess.PIPE)
    # エラーがあったら出力する
    for line in p.stderr:
        print line,
