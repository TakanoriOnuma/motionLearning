# -*- coding: utf-8 -*-

import os
import cv2
import json
import glob
import numpy
import shutil

from sklearn.cluster import KMeans

# 自作ライブラリのパスを設定してから読み込む
import sys
sys.path.append('C:\Python27\motionLearning')
import mylib.util

VIEW_LIST = ['view45,45', 'view0,0', 'view90,0', 'view90,90']
# 記録クラス
class Reporter:
    # 初期化
    def __init__(self, gpDirName, pyDirName):
        self.gpDirName = gpDirName
        self.pyDirName = pyDirName

    # 記録対象のディレクトリ名を設定する
    def setReportDirName(self, dirName):
        self.dirName = dirName
        self.prop    = json.load(open(self.dirName + '/property.json', 'r'))

    # 学習エラーをグラフに表示する
    def drawError(self):
        print '- draw error.'
        arg = "path='{}'; limit={}".format(self.dirName, self.prop['TRAIN_NUM'])
        exeName = self.gpDirName + '/error.gp'
        mylib.util.doGnuplot(arg, exeName)

    # testデータを統合する
    def integrateTestSwing(self):
        print '- integrate test swing.'
        srcDirName  = '{}/middle/{}'.format(self.dirName, 'swing')
        destDirName = '{}/middle/{}'.format(self.dirName, 'test')
        mylib.util.mkdir(destDirName)
        for num in mylib.util.logrange(0, self.prop['TRAIN_NUM']):
            self.__integrateSwingData(srcDirName, destDirName, num, self.prop['TRAIN_SWING'])

    # swingデータの統合
    def __integrateSwingData(self, srcDirName, destDirName, num, exceptNumList):
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

    # 中間層のニューロンの出力をグラフに描画する
    def drawOutputInMiddleLayer(self):
        print '- draw output of middle layer.'
        mylib.util.mkdir(self.dirName + '/outNeuron')
        # 各ディレクトリの記録
        for trainType in ['train', 'test']:
            print '  - {} directory'.format(trainType)
            mylib.util.mkdir('{}/outNeuron/{}'.format(self.dirName, trainType))
            if self.prop['featureNum'] == 3:
                for viewDir in VIEW_LIST:
                    mylib.util.mkdir('{}/outNeuron/{}/{}'.format(self.dirName, trainType, viewDir))
            print '    - draw'
            arg = "path='{}'; subDirName='{}'; limit={}".format(self.dirName, trainType, self.prop['TRAIN_NUM'])
            exeName = self.gpDirName + '/{}DNeuron.gp'.format(self.prop['featureNum'])
            mylib.util.doGnuplot(arg, exeName)

            if self.prop['featureNum'] == 3:
                print '    - concat'
                mylib.util.mkdir('{}/outNeuron/{}/concat'.format(self.dirName, trainType))
                self.__saveConcatImages('{}/outNeuron/{}'.format(self.dirName, trainType), self.prop['TRAIN_NUM'])
        # 各swingの記録
        print '  - each swing'
        mylib.util.mkdir('{}/outNeuron/swing'.format(self.dirName))
        for swingNum in range(self.prop['SWING_NUM']):
            mylib.util.mkdir('{}/outNeuron/swing/swing{}'.format(self.dirName, swingNum))
            if self.prop['featureNum'] == 3:
                for viewDir in VIEW_LIST:
                    mylib.util.mkdir('{}/outNeuron/swing/swing{}/{}'.format(self.dirName, swingNum, viewDir))
        print '    - draw'
        arg = "path='{}'; subDirName='swing'; limit={}; swingLimit={}" \
                .format(self.dirName, self.prop['TRAIN_NUM'], self.prop['SWING_NUM'])
        exeName = self.gpDirName + '/{}DNeuron_swing.gp'.format(self.prop['featureNum'])
        mylib.util.doGnuplot(arg, exeName)
        if self.prop['featureNum'] == 3:
            print '    - concat'
            for swingNum in range(self.prop['SWING_NUM']):
                rootDirName = '{}/outNeuron/swing/swing{}'.format(self.dirName, swingNum)
                mylib.util.mkdir(rootDirName + '/concat')
                self.__saveConcatImages(rootDirName, self.prop['TRAIN_NUM'])
        # 学習回数ごとのフォルダを作成する
        print '    - copy'
        rootDirName = '{}/outNeuron/swing'.format(self.dirName)
        mylib.util.mkdir(rootDirName + '/swing_all')
        for n in mylib.util.logrange(0, self.prop['TRAIN_NUM']):
            mylib.util.mkdir('{}/swing_all/{}'.format(rootDirName, n))
            for swingNum in range(self.prop['SWING_NUM']):
                if self.prop['featureNum'] == 2:
                    src = '{}/swing{}/out_neuron{}.png'.format(rootDirName, swingNum, n)
                else:
                    src = '{}/swing{}/concat/out_neuron{}.png'.format(rootDirName, swingNum, n)
                dst = '{}/swing_all/{}/out_neuron{}.png'.format(rootDirName, n, swingNum)
                shutil.copy(src, dst)

    # 画像の結合
    def __saveConcatImages(self, root, limit):
        for n in mylib.util.logrange(0, limit):
            imgs = []
            for subDirName in VIEW_LIST:
                fileName = "{0}/{1}/out_neuron{2}_{1}.png".format(root, subDirName, n)
                imgs.append(cv2.imread(fileName))
            image = mylib.image.concat(imgs[0], imgs[1], imgs[2], imgs[3])
            cv2.imwrite("{0}/concat/out_neuron{1}.png".format(root, n), image)

    # 特徴層に直接入力した際の出力画像を記録する
    def saveOutputOfDirectActivation(self):
        print '- draw output of direct activation.'
        mylib.util.doPython(self.pyDirName + '/directActivate.py', self.dirName)

    # クラスタリングの結果を保存する
    def saveClustering(self, clusterNum):
        print '- clustering.'
        # 1スイングの点数を30点で統一する
        print '  - normalize swing points.'
        for swingNum in range(self.prop['SWING_NUM']):
            rootDirName = '{}/middle/swing/swing{}'.format(self.dirName, swingNum)
            for i in mylib.util.logrange(0, self.prop['TRAIN_NUM']):
                fileName = '{}/middle{}.dat'.format(rootDirName, i)
                points = self.__getPoints(fileName)
                normPoints = self.__normalizePoints(points, 30)
                mylib.util.mkdir('{}/normalize'.format(rootDirName))
                fileName = '{}/normalize/middle{}.dat'.format(rootDirName, i)
                self.__savePoints(fileName, normPoints)
        # 各スイングを1次元情報に変換する
        swings = []
        for swingNum in range(self.prop['SWING_NUM']):
            rootDirName = '{}/middle/swing/swing{}'.format(self.dirName, swingNum)
            fileName = '{}/normalize/middle{}.dat'.format(rootDirName, self.prop['TRAIN_NUM'])
            points = numpy.array(self.__getPoints(fileName))
            swings.append(numpy.reshape(points, points.shape[0] * points.shape[1]))
        swings = numpy.array(swings)
        # k-Mean法を使ってクラスタリングする
        print '  - kmeans.'
        labels = KMeans(n_clusters=clusterNum).fit_predict(swings)
        # 結果を基に分類する
        print '  - distribute by the result.'
        rootDirName = '{}/clustering'.format(self.dirName)
        # 実行の度に結果が変わるためフォルダごと消して初期化する
        if os.path.exists(rootDirName):
            shutil.rmtree(rootDirName)
        mylib.util.mkdir(rootDirName)
        for i in range(clusterNum):
            createDirName = '{}/{}'.format(rootDirName, i)
            mylib.util.mkdir(createDirName)
        swingDirName = '{}/outNeuron/swing/swing_all/{}'.format(self.dirName, self.prop['TRAIN_NUM'])
        for swingNum in range(self.prop['SWING_NUM']):
            src = '{}/out_neuron{}.png'.format(swingDirName, swingNum)
            dst = '{}/{}/out_neuron{}.png'.format(rootDirName, labels[swingNum], swingNum)
            shutil.copy(src, dst)
        # gifアニメもコピーしてくる
        IMG_DIR = 'C:\Python27\motionLearning\learning\IMAGES'
        imgDir  = '{}/{}/{}'.format(IMG_DIR, self.prop['IMG_DIR'], self.prop['DATA_TYPE'])
        for swingNum in range(self.prop['SWING_NUM']):
            src = '{}/swing/{}/ani2.gif'.format(imgDir, swingNum)
            dst = '{}/clustering/{}/ani{}.gif'.format(self.dirName, labels[swingNum], swingNum)
            shutil.copy(src, dst)
        # クラスに分類する
        clusters = [[] for i in range(clusterNum)]
        for swingNum in range(self.prop['SWING_NUM']):
            clusters[labels[swingNum]].append(swingNum)
        # HTMLページを作成する
        fHtml = open('{}/clustering/clustering.html'.format(self.dirName), 'w')
        fHtml.write('<table border="1">\n')
        for classNum in range(clusterNum):
            fHtml.write('  <tr align="center">\n')
            fHtml.write('    ')
            fHtml.write('<td>{}</td>'.format(classNum))
            for swingNum in clusters[classNum]:
                fHtml.write('<td><img src="{0}/ani{1}.gif"><br>{1}</td>'.format(classNum, swingNum))
            fHtml.write('\n')
            fHtml.write('  </tr>\n')
        fHtml.write('</table>')
        fHtml.close()
        

    # 点情報を取得する
    def __getPoints(self, fileName, separator='\t'):
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
    def __normalizePoints(self, points, numPoint):
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
    def __savePoints(self, fileName, points, separator='\t'):
        fPoints = open(fileName, 'w')
        fPoints.write('# ' + separator.join(['middle{}'.format(i) for i in range(len(points[0]))]) + '\n')
        for i in range(len(points)):
            fPoints.write(separator.join([str(pt) for pt in points[i]]) + '\n')
        fPoints.close() 
