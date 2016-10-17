# -*- coding: utf-8 -*-
import os
import glob
import cv2
import shutil

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

    # testデータを統合する
    print '- integrate test swing.'
    srcDirName  = '{}/{}/middle/{}'.format(ROOT, dirName, 'swing')
    destDirName = '{}/{}/middle/{}'.format(ROOT, dirName, 'test')
    mylib.util.mkdir(destDirName)
    integrateSwingData(srcDirName, destDirName, 0, range(41, 61))
    for num in mylib.util.logrange(1, prop['TRAIN_NUM']):
        integrateSwingData(srcDirName, destDirName, num, range(41, 61))           

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
