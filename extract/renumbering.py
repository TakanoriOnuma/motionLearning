# -*- coding: utf-8 -*-
import glob
import re
import os

# 検索された中で最小値と最大値を返す
def getMinAndMax(fileNames, p):
    minValue = 100000
    maxValue = 0
    for fileName in fileNames:
        m = p.search(fileName)
        value = int(m.group(1))
        if value < minValue:
            minValue = value
        elif value > maxValue:
            maxValue = value

    return minValue, maxValue

# startからendを0からend - startまでの番号に変える
def fileNameChange(formatStr, start, end):
    i = 0
    for idx in range(start, end + 1):
        old = formatStr.format(idx)
        new = formatStr.format(i)
        print old, new
        os.rename(old, new)
        i += 1


ROOT = 'images5/normal'

# trainフォルダを見る
searchFileName = '{}/train/*.png'.format(ROOT)
p = re.compile('img([0-9]+).png')
start, end = getMinAndMax(glob.glob(searchFileName), p)
fileNameChange(ROOT + '/train/img{}.png', start, end)

# testフォルダを見る
rootDir = '{}/test/*[0-9]'.format(ROOT)
for dirName in glob.glob(rootDir):
    searchFileName = '{}/*.png'.format(dirName)
    start, end = getMinAndMax(glob.glob(searchFileName), p)
    print start, end
    fileNameChange(dirName + '/img{}.png', start, end)


