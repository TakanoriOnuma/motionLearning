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
