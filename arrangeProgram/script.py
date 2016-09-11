# -*- coding: utf-8 -*-
import os
import glob
import cv2

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
    saveConcatImage(root, 1, viewList)
    step = 1
    n = 1
    while n < limit:
        for cnt in range(9):
            n += step
            saveConcatImage(root, n, viewList)
        step *= 10

ROOT = os.getcwd()
VIEW_LIST = ['view45,45', 'view0,0', 'view90,0', 'view90,90']
for dirName in glob.glob('part*'):
    print dirName
    # propertyの読み込み
    prop = mylib.property.readProperty('{}/{}/property.txt'.format(ROOT, dirName))
    
    # 学習エラーをグラフに表示する
    print '- draw error.'
    arg = "path='{}/{}'; limit={}".format(ROOT, dirName, prop['TRAIN_NUM'])
    exeName = 'gnuplot/error.gp'
    mylib.util.doGnuplot(arg, exeName)

    # 中間層のニューロンの出力をグラフに表示する
    print '- draw output of middle layer.'
    mylib.util.mkdir('{}/{}/outNeuron'.format(ROOT, dirName))
    # trainディレクトリの記録
    print '  - train directory'
    mylib.util.mkdir('{}/{}/outNeuron/train'.format(ROOT, dirName))
    for viewDir in VIEW_LIST:
        mylib.util.mkdir('{}/{}/outNeuron/train/{}'.format(ROOT, dirName, viewDir))
    arg = "path='{}/{}'; subDirName='train'; limit={}".format(ROOT, dirName, prop['TRAIN_NUM'])
    exeName = 'gnuplot/3DNeuron.gp'
    mylib.util.doGnuplot(arg, exeName)
    print '    - concat'
    mylib.util.mkdir('{}/{}/outNeuron/train/concat'.format(ROOT, dirName))
    saveConcatImages('{}/{}/outNeuron/train'.format(ROOT, dirName), prop['TRAIN_NUM'], VIEW_LIST)
    # testディレクトリの記録
    mylib.util.mkdir('{}/{}/outNeuron/test'.format(ROOT, dirName))
    for swingNum in range(prop['SWING_NUM']):
        mylib.util.mkdir('{}/{}/outNeuron/test/{}'.format(ROOT, dirName, swingNum))
        for viewDir in VIEW_LIST:
            mylib.util.mkdir('{}/{}/outNeuron/test/{}/{}'.format(ROOT, dirName, swingNum, viewDir))
    print '  - test directory'
    arg = "path='{}/{}'; subDirName='test'; limit={}; swingLimit={}" \
            .format(ROOT, dirName, prop['TRAIN_NUM'], prop['SWING_NUM'])
    exeName = 'gnuplot/3DNeuron_swing.gp'
    mylib.util.doGnuplot(arg, exeName)
    print '    - concat'
    for swingNum in range(prop['SWING_NUM']):
        rootDirName = '{}/{}/outNeuron/test/{}'.format(ROOT, dirName, swingNum)
        mylib.util.mkdir(rootDirName + '/concat')
        saveConcatImages(rootDirName, prop['TRAIN_NUM'], VIEW_LIST)
