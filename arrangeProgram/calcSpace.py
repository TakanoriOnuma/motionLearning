# -*- coding: utf-8 -*-

import sys
import json
import numpy as np

PATH = 'C:/Python27/motionLearning/'
sys.path.append(PATH)
import mylib

# 点情報を取得する
def getMiddlePoints(dirName, maxSwingNum, epoch):
    points = []
    for swingNum in range(maxSwingNum):
        fileName = dirName + '/swing{}/middle{}.dat'.format(swingNum, epoch)
        pts = mylib.point.getPoints(fileName)
        points.append(np.array(pts[:-1]))
    return np.vstack(points)

# ヒストグラムテーブルの作成
def makeDistributionTable(dataList, div = 1):
    scale = 1.0 / div
    table = np.zeros([div] * len(dataList[0]))
    for data in dataList:
        idx = [int(x / scale) for x in data]
        table[tuple(idx)] += 1
    return table

# データのばらつきを記録
def saveDistribution(fileName, dataList, divs):
    fDist = open(fileName, 'w')
    columns = ['div', 'std', 'free_blocks']
    fDist.write('# ' + '\t'.join([str(x) for x in columns]) + '\n')
    for div in divs:
        table = makeDistributionTable(dataList, div)
        params = [div, table.std(), len(table[table==0])];
        fDist.write('\t'.join([str(x) for x in params]) + '\n')
    fDist.close()

ROOT = sys.argv[1] if len(sys.argv) == 2 else 'C:/Python27/motionLearning/arrangeProgram/part4'

prop = json.load(open(ROOT + '/property.json', 'r'))

swingDirName = ROOT + '/middle/swing'

distDirName = ROOT + '/middle/distribution'
mylib.util.mkdir(distDirName)

divs = [10, 20, 30]
dists = {}
for div in divs:
    dists[str(div)] = []

for epoch in mylib.util.logrange(0, prop['TRAIN_NUM']):
    points = getMiddlePoints(swingDirName, prop['SWING_NUM'], epoch)
    for div in divs:
        table  = makeDistributionTable(points, div)
        params = [epoch, table.std(), len(table[table==0])]
        dists[str(div)].append(params)

for div in divs:
    dist = dists[str(div)]
    filePath = distDirName + '/distribution_{}.dat'.format(div)
    f = open(filePath, 'w')
    f.write('# epoch\t' + 'distribution\t' + 'free_space\n')
    for params in dist:
        f.write('\t'.join([str(x) for x in params]) + '\n')
    f.close()

    fileName = filePath.split('/')[-1]
    arg = "path='{0}'; fileName='{1}'; titleName='{1}'; limit={2}" \
        .format(distDirName, fileName.split('.')[0], prop['TRAIN_NUM'])
    exeName = 'gnuplot/distribution.gp'
    mylib.util.doGnuplot(arg, exeName)
