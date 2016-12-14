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

    # 1スイングの点数を統一する
    def normalizeSwingPoints(self, pointNum):
        print '- normalize swing points.'
        for swingNum in range(self.prop['SWING_NUM']):
            rootDirName = '{}/middle/swing/swing{}'.format(self.dirName, swingNum)
            for i in mylib.util.logrange(0, self.prop['TRAIN_NUM']):
                fileName = '{}/middle{}.dat'.format(rootDirName, i)
                points = mylib.point.getPoints(fileName)
                normPoints = mylib.point.normalizePoints(points, pointNum)
                mylib.util.mkdir('{}/normalize'.format(rootDirName))
                fileName = '{}/normalize/middle{}.dat'.format(rootDirName, i)
                mylib.point.savePoints(fileName, normPoints)

    # 各スイングを1次元情報として取得する
    def __getSwings(self, maxSwingNum, epoch):
        swings = []
        for swingNum in range(maxSwingNum):
            rootDirName = '{}/middle/swing/swing{}'.format(self.dirName, swingNum)
            fileName = '{}/normalize/middle{}.dat'.format(rootDirName, epoch)
            points = numpy.array(mylib.point.getPoints(fileName))
            # 点の数と次数を記録しておく
            pointNum = points.shape[0]
            dim = points.shape[1]
            swings.append(numpy.reshape(points, points.shape[0] * points.shape[1]))
        swings = numpy.array(swings)
        return swings, pointNum, dim

    # クラスタリングをする
    def __clustering(self, dats, clusterNum):
        labels = KMeans(n_clusters=clusterNum).fit_predict(dats)
        # 各クラスにswing番号を配布する
        clusters = [[] for i in range(clusterNum)]
        for idx in range(len(dats)):
            clusters[labels[idx]].append(idx)
        return clusters

    # クラスタリングの結果を保存する
    def saveClustering(self, clusterNum):
        print '- clustering.'
        # 各スイングを1次元情報に変換する
        swings, pointNum, dim = self.__getSwings(self.prop['SWING_NUM'], self.prop['TRAIN_NUM'])
        # k-Mean法を使ってクラスタリングする
        print '  - kmeans.'
        clusters = self.__clustering(swings, clusterNum)
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
        # グラフとgifアニメのフォルダ先を指定する
        swingDirName = '{}/outNeuron/swing/swing_all/{}'.format(self.dirName, self.prop['TRAIN_NUM'])
        IMG_DIR = 'C:\Python27\motionLearning\learning\IMAGES'
        imgDir  = '{}/{}/{}'.format(IMG_DIR, self.prop['IMG_DIR'], self.prop['DATA_TYPE'])
        for classNum in range(clusterNum):
            for swingNum in clusters[classNum]:
                # グラフのコピー
                src = '{}/out_neuron{}.png'.format(swingDirName, swingNum)
                dst = '{}/clustering/{}/out_neuron{}.png'.format(self.dirName, classNum, swingNum)
                shutil.copy(src, dst)
                # gifアニメのコピー
                src = '{}/swing/{}/ani2.gif'.format(imgDir, swingNum)
                dst = '{}/clustering/{}/ani{}.gif'.format(self.dirName, classNum, swingNum)
                shutil.copy(src, dst)
        # 各クラスの重心を記録する
        self.__saveMeanSwings(clusters, swings, (pointNum, dim))
        # 重心情報からアニメーションを作成する
        mylib.util.doPython(self.pyDirName + '/directInput.py', self.dirName, str(clusterNum))
        # 各クラスの重心を記録する
        self.__saveMeanImages(clusters, imgDir)
        # 重心の画像から軌跡を作成する
        mylib.util.doPython(self.pyDirName + '/makeLocus.py', self.dirName, str(clusterNum))
        # クラス同士で差分をとる
        mylib.util.doPython(self.pyDirName + '/imgDifference.py', self.dirName, str(clusterNum))
        # クラスタリングの結果をHTMLに出力する
        self.__saveClusteringToHtml(clusters)       

    # 各クラスの重心を記録する
    def __saveMeanSwings(self, clusters, swings, size):
        for classNum in range(len(clusters)):
            # クラスに属するスイング集合を取得
            classSwings = numpy.array(swings[clusters[classNum]])
            # スイング集合の平均を取る
            meanSwing = classSwings.mean(axis=0).reshape(size)
            fileName = '{}/clustering/{}/mean.dat'.format(self.dirName, classNum)
            mylib.point.savePoints(fileName, meanSwing)
            # グラフにする
            titleName = 'out_neuron{}_neuron_swing/mean{}'.format(self.prop['TRAIN_NUM'], classNum)
            mylib.point.drawPoints('{}/clustering/{}'.format(self.dirName, classNum), 'mean', titleName)

    # 各クラスの重心を記録する（入力画像版）
    def __saveMeanImages(self, clusters, imgDir):
        # データを取得する
        num = len(glob.glob(imgDir + '/swing/0/norm/*.png'))
        for classNum in range(len(clusters)):
            classImgs = []
            for swingNum in clusters[classNum]:
                imgs = []
                for i in range(num):
                    fileName = imgDir + '/swing/{}/norm/img{}.png'.format(swingNum, i)
                    img = cv2.imread(fileName)
                    height = img.shape[0]
                    width  = img.shape[1]
                    imgs.append(mylib.image.makeInputData(img, 'data', 0))
                classImgs.append(imgs)
            meanImgs = numpy.array(classImgs).mean(axis=0)
            mylib.util.mkdir('{}/clustering/{}/norm'.format(self.dirName, classNum))
            imgFileNames = []
            for i in range(len(meanImgs)):
                img = mylib.image.makeOutputData(meanImgs[i], height, width)
                fileName = '{}/clustering/{}/norm/img{}.png'.format(self.dirName, classNum, i)
                cv2.imwrite(fileName, img)
                imgFileNames.append(fileName)
            fileName = '{}/clustering/{}/norm/ani.gif'.format(self.dirName, classNum)
            mylib.image.makeGifAnime(imgFileNames, 3, fileName)

    # クラスタリングの結果をHTMLにまとめる
    def __saveClusteringToHtml(self, clusters):
        clusterNum = len(clusters)
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
        # error量を取得する
        fErrors = open('{}/output/swing/error_prop.dat'.format(self.dirName), 'r')
        errors = []
        for line in fErrors:
            if line[0] == '#':
                continue
            swingNum, error = line[:-1].split('\t')
            errors.append(error)
        # 恒等写像で得られた画像を使ったHTMLページを作成する
        fHtml = open('{}/clustering/clustering2.html'.format(self.dirName), 'w')
        fHtml.write('<table border="1">\n')
        for classNum in range(clusterNum):
            fHtml.write('  <tr align="center">\n')
            fHtml.write('    ')
            fHtml.write('<td>{}</td>'.format(classNum))
            fHtml.write('<td><img src="{}/clustering/{}/mean/ani.gif"><br><br><br></td>'.format(self.dirName, classNum))
            for swingNum in clusters[classNum]:
                fHtml.write('<td><img src="../output/swing/swing{0}/ani2.gif"><br>{0}<br>{1}</td>'.format(swingNum, errors[swingNum]))
            fHtml.write('\n')
            fHtml.write('  </tr>\n')
        fHtml.write('</table>')
        fHtml.close()
        # グラフの結果をHTMLページにまとめる
        fHtml = open('{}/clustering/clustering_graph.html'.format(self.dirName), 'w')
        fHtml.write('<table border="1">\n')
        for classNum in range(clusterNum):
            fHtml.write('  <tr align="center">\n')
            fHtml.write('    ')
            fHtml.write('<td>{}<br>({})</td>'.format(classNum, len(clusters[classNum])))
            fHtml.write('<td><img src="{}/mean.png" width="300" height="200"></td>'.format(classNum))
            for swingNum in clusters[classNum]:
                fHtml.write('<td><img src="{0}/out_neuron{1}.png" width="300" height="200"></td>'.format(classNum, swingNum))
            fHtml.write('\n')
            fHtml.write('  </tr>\n')
        fHtml.write('</table>')
        fHtml.close()
        # 平均情報をHTMLページにまとめる
        fHtml = open('{}/clustering/mean.html'.format(self.dirName), 'w')
        fHtml.write('<table border="1">\n')
        for classNum in range(clusterNum):
            fHtml.write('  <tr align="center">\n')
            fHtml.write('    ')
            fHtml.write('<td>{}<br>({})</td>'.format(classNum, len(clusters[classNum])))
            fHtml.write('<td><img src="{}/mean.png" width="300" height="200"></td>'.format(classNum))
            fHtml.write('<td><img src="{}/mean/ani.gif" width="200" height="200"></td>'.format(classNum))
            fHtml.write('<td><img src="{}/norm/ani.gif" width="200" height="200"></td>'.format(classNum))
            fHtml.write('<td><img src="{}/norm/norm.png" width="300" height="200"></td>'.format(classNum))
            fHtml.write('\n')
            fHtml.write('  </tr>\n')
        fHtml.write('</table>')
        fHtml.close()
        # クラス同士の差分をHTMLページにまとめる
        fHtml = open('{}/clustering/diff/diff.html'.format(self.dirName), 'w')
        fHtml.write('<table border="1">\n')
        fHtml.write('  <tr align="center">')
        for classNum in range(clusterNum):
            fHtml.write('<td>')
            fHtml.write('<img src="../{0}/norm/ani.gif" width="200" height="200"><br>{0}<br>({1})'.format(classNum, len(clusters[classNum])))
            fHtml.write('</td>\n')
        fHtml.write('  </tr>\n')
        fHtml.write('</table>\n')
        fHtml.write('<br>\n')
        
        fHtml.write('<table border="1">\n')
        fHtml.write('  <tr align="center"><td></td>' + ''.join(['<td>{}</td>'.format(i) for i in range(clusterNum)]) + '</tr>\n')
        for srcClassNum in range(clusterNum):
            fHtml.write('  <tr align="center">\n')
            fHtml.write('    <td>{}</td>'.format(srcClassNum))
            for compClassNum in range(clusterNum):
                fHtml.write('<td>')
                if srcClassNum != compClassNum:
                    fHtml.write('<img src="{0}_{1}/ani.gif" width="200" height="200"><br>{0} vs. {1}<br>'.format(srcClassNum, compClassNum))
                fHtml.write('</td>\n')
            fHtml.write('  </tr>\n')
        fHtml.write('</table>')
        fHtml.close()
        

    # 恒等写像した出力画像を保存する
    def saveIdentityMapping(self):
        print '- save identity mapping.'
        mylib.util.doPython(self.pyDirName + '/saveIdentityMapping.py', self.dirName)
        print '  - make animation.'
        mylib.util.doPython(self.pyDirName + '/makeGifAnime.py', self.dirName);
