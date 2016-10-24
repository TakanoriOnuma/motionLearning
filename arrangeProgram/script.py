# -*- coding: utf-8 -*-
import os
import numpy
import json
import glob
import cv2
import shutil

from sklearn.cluster import KMeans

# 自作ライブラリのパスを設定してから読み込む
import sys
sys.path.append('C:\Python27\motionLearning')
import mylib

# viewListからそれぞれ読みだして、それを結合して保存する
def saveConcatImage(root, n, viewList):
    imgs = []
    for subDirName in viewList:
        fileName = "{0}/{1}/out_neuron{2}_{1}.png".format(root, subDirName, n)
        imgs.append(cv2.imread(fileName))
    image = mylib.image.concat(imgs[0], imgs[1], imgs[2], imgs[3])
    cv2.imwrite("{0}/concat/out_neuron{1}.png".format(root, n), image)

# 対数スケールで保存していく
def saveConcatImages(root, limit, viewList):
    saveConcatImage(root, 0, viewList)
    for n in mylib.util.logrange(1, limit):
        saveConcatImage(root, n, viewList)

# swingデータの統合
def integrateSwingData(srcDirName, destDirName, num, exceptNumList):
    fConcatMiddle = open('{}/middle{}.dat'.format(destDirName, num), 'w')
    for i in range(len(glob.glob('{}/swing[0-9]*'.format(srcDirName)))):
        if i in exceptNumList:
            continue
        fConcatMiddle.write('# {}\n'.format(i))
        fMiddle = open('{}/swing{}/middle{}.dat'.format(srcDirName, i, num), 'r')
        for line in fMiddle:
            fConcatMiddle.write(line)
        fMiddle.close()
    fConcatMiddle.close()

# 点情報を取得する
def getPoints(fileName, separator='\t'):
    fPoints = open(fileName, 'r')
    points = []
    for line in fPoints:
        if line[0] == '#':
            continue
        pts = line.split(separator)
        for i in range(len(pts) - 1):
            pts[i] = float(pts[i])
        pts[-1] = float(pts[-1][:-1])
        points.append(pts)
    return points

# 点情報をnumPointで規格化
def normalizePoints(points, numPoint):
    step = float(len(points) - 1) / (numPoint - 1)
    normPoints = []
    for i in range(numPoint):
        pt      = i * step
        intPt   = int(pt)
        floatPt = pt - intPt
        # 小数点がない場合は整数の点をそのまま取ってくる
        if floatPt <= 0.1e-10:
            pts = points[intPt]
        # 小数もある場合は内分点の計算をする
        else:
            pts = []
            for j in range(len(points[intPt])):
                pts.append((1 - floatPt) * points[intPt][j] + floatPt * points[intPt + 1][j])
        normPoints.append(pts)
    return normPoints

# 点情報を保存する
def savePoints(fileName, points, separator='\t'):
    fPoints = open(fileName, 'w')
    fPoints.write('# ' + separator.join(['middle{}'.format(i) for i in range(len(points[0]))]) + '\n')
    for i in range(len(points)):
        fPoints.write(separator.join([str(pt) for pt in points[i]]) + '\n')
    fPoints.close()

ROOT = os.getcwd()
VIEW_LIST = ['view45,45', 'view0,0', 'view90,0', 'view90,90']
for dirName in glob.glob('part*'):
    print dirName
    # propertyの読み込み
    prop = json.load(open('{}/{}/property.json'.format(ROOT, dirName), 'r'))

    # 学習エラーをグラフに表示する
    print '- draw error.'
    arg = "path='{}/{}'; limit={}".format(ROOT, dirName, prop['TRAIN_NUM'])
    exeName = 'gnuplot/error.gp'
    mylib.util.doGnuplot(arg, exeName)

    # testデータを統合する
    print '- integrate test swing.'
    srcDirName  = '{}/{}/middle/{}'.format(ROOT, dirName, 'swing')
    destDirName = '{}/{}/middle/{}'.format(ROOT, dirName, 'test')
    mylib.util.mkdir(destDirName)
    integrateSwingData(srcDirName, destDirName, 0, prop['TRAIN_SWING'])
    for num in mylib.util.logrange(1, prop['TRAIN_NUM']):
        integrateSwingData(srcDirName, destDirName, num, prop['TRAIN_SWING'])    

    # 中間層のニューロンの出力をグラフに表示する
    print '- draw output of middle layer.'
    mylib.util.mkdir('{}/{}/outNeuron'.format(ROOT, dirName))
    # 各ディレクトリの記録
    for trainType in ['train', 'test']:
        print '  - {} directory'.format(trainType)
        mylib.util.mkdir('{}/{}/outNeuron/{}'.format(ROOT, dirName, trainType))
        for viewDir in VIEW_LIST:
            mylib.util.mkdir('{}/{}/outNeuron/{}/{}'.format(ROOT, dirName, trainType, viewDir))
        print '    - draw'
        arg = "path='{}/{}'; subDirName='{}'; limit={}".format(ROOT, dirName, trainType, prop['TRAIN_NUM'])
        exeName = 'gnuplot/3DNeuron.gp'
        mylib.util.doGnuplot(arg, exeName)
        print '    - concat'
        mylib.util.mkdir('{}/{}/outNeuron/{}/concat'.format(ROOT, dirName, trainType))
        saveConcatImages('{}/{}/outNeuron/{}'.format(ROOT, dirName, trainType), prop['TRAIN_NUM'], VIEW_LIST)
    # 各swingの記録
    print '  - each swing'
    mylib.util.mkdir('{}/{}/outNeuron/swing'.format(ROOT, dirName))
    for swingNum in range(prop['SWING_NUM']):
        mylib.util.mkdir('{}/{}/outNeuron/swing/swing{}'.format(ROOT, dirName, swingNum))
        for viewDir in VIEW_LIST:
            mylib.util.mkdir('{}/{}/outNeuron/swing/swing{}/{}'.format(ROOT, dirName, swingNum, viewDir))
    print '    - draw'
    arg = "path='{}/{}'; subDirName='swing'; limit={}; swingLimit={}" \
            .format(ROOT, dirName, prop['TRAIN_NUM'], prop['SWING_NUM'])
    exeName = 'gnuplot/3DNeuron_swing.gp'
    mylib.util.doGnuplot(arg, exeName)
    print '    - concat'
    for swingNum in range(prop['SWING_NUM']):
        rootDirName = '{}/{}/outNeuron/swing/swing{}'.format(ROOT, dirName, swingNum)
        mylib.util.mkdir(rootDirName + '/concat')
        saveConcatImages(rootDirName, prop['TRAIN_NUM'], VIEW_LIST)
    print '    - copy'
    rootDirName = '{}/{}/outNeuron/swing'.format(ROOT, dirName)
    mylib.util.mkdir(rootDirName + '/swing_all')
    for n in mylib.util.logrange(1, prop['TRAIN_NUM']):
        mylib.util.mkdir('{}/swing_all/{}'.format(rootDirName, n))
        for swingNum in range(prop['SWING_NUM']):
            src = '{}/swing{}/concat/out_neuron{}.png'.format(rootDirName, swingNum, n)
            dst = '{}/swing_all/{}/out_neuron{}.png'.format(rootDirName, n, swingNum)
            shutil.copy(src, dst)

    # 特徴層に直接入力した際の出力画像を記録する
    print '- draw output of direct activation.'
    mylib.util.doPython('directActivate.py', '{}/{}'.format(ROOT, dirName))

    # クラスタリングする
    print '- clustering.'
    mylib.util.mkdir('{}/{}/clustering'.format(ROOT, dirName))
    # 1スイングの点数を30点で統一する
    print '  - normalize swing points.'
    for swingNum in range(prop['SWING_NUM']):
        rootDirName = '{}/{}/middle/swing/swing{}'.format(ROOT, dirName, swingNum)
        for i in mylib.util.logrange(0, prop['TRAIN_NUM']):
            fileName = '{}/middle{}.dat'.format(rootDirName, i)
            points = getPoints(fileName)
            normPoints = normalizePoints(points, 30)
            mylib.util.mkdir('{}/normalize'.format(rootDirName))
            fileName = '{}/normalize/middle{}.dat'.format(rootDirName, i)
            savePoints(fileName, normPoints)
    # 各スイングを1次元情報に変換する
    swings = []
    for swingNum in range(prop['SWING_NUM']):
        rootDirName = '{}/{}/middle/swing/swing{}'.format(ROOT, dirName, swingNum)
        fileName = '{}/normalize/middle{}.dat'.format(rootDirName, prop['TRAIN_NUM'])
        points = numpy.array(getPoints(fileName))
        swings.append(numpy.reshape(points, points.shape[0] * points.shape[1]))
    swings = numpy.array(swings)
    # k-Mean法を使ってクラスタリングする
    print '  - kmeans.'
    CLUSTER_NUM = 3
    labels = KMeans(n_clusters=CLUSTER_NUM).fit_predict(swings)
    # 結果を基に分類する
    print '  - distribute by the result.'
    rootDirName = '{}/{}/clustering'.format(ROOT, dirName)
    for i in range(CLUSTER_NUM):
        createDirName = '{}/{}'.format(rootDirName, i)
        # 実行の度に結果が変わるためフォルダごと消して初期化する
        if os.path.exists(createDirName):
            shutil.rmtree(createDirName)
        mylib.util.mkdir(createDirName)
    swingDirName = '{}/{}/outNeuron/swing/swing_all/{}'.format(ROOT, dirName, prop['TRAIN_NUM'])
    for swingNum in range(prop['SWING_NUM']):
        src = '{}/out_neuron{}.png'.format(swingDirName, swingNum)
        dst = '{}/{}/out_neuron{}.png'.format(rootDirName, labels[swingNum], swingNum)
        shutil.copy(src, dst)
    
