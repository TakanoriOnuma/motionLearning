# -*- coding: utf-8 -*-
import os
import glob

# 自作ライブラリのパスを設定してから読み込む
import sys
sys.path.append('C:\Python27\motionLearning')
import mylib

ROOT = os.getcwd()
VIEW_LIST = ['view0,0', 'view90,0', 'view90,90', 'view45,45']
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
