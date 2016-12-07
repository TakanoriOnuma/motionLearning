# -*- coding: utf-8 -*-
import sys, os, cv2
import numpy
import json
import glob

sys.path.append('C:\Python27\motionLearning')
import mylib

# 実行ディレクトリを設定する
ROOT = sys.argv[1] if len(sys.argv) == 3 else 'C:/Python27/motionLearning/arrangeProgram/part4'
clusterNum = int(sys.argv[2]) if len(sys.argv) == 3 else 3

# 各クラスの差分画像を記録する
mylib.util.mkdir(ROOT + '/clustering/diff')
for srcClassNum in range(clusterNum):
    num = len(glob.glob(ROOT + '/clustering/{}/norm/img*.png'.format(srcClassNum)))

    for compClassNum in range(clusterNum):
        # 同じクラスの場合は省略
        if srcClassNum == compClassNum:
            continue

        mylib.util.mkdir(ROOT + '/clustering/diff/{}_{}'.format(srcClassNum, compClassNum))
        fileNames = []
        for idx in range(num):    
            srcImg  = cv2.imread(ROOT + '/clustering/{}/norm/img{}.png'.format(srcClassNum, idx))
            srcImg  = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
            compImg = cv2.imread(ROOT + '/clustering/{}/norm/img{}.png'.format(compClassNum, idx))
            compImg = cv2.cvtColor(compImg, cv2.COLOR_BGR2GRAY)

            diff = srcImg.astype(numpy.int32) - compImg

            diffImg = numpy.zeros((srcImg.shape[0], srcImg.shape[1], 3), numpy.uint8)
            THRESHOLD = 20
            for i in range(srcImg.shape[0]):
                for j in range(srcImg.shape[1]):
                    if diff[i][j] < -THRESHOLD:
                        diffImg[i][j][0] = min(255, 5 * (-diff[i][j]))
                    elif diff[i][j] > THRESHOLD:
                        diffImg[i][j][2] = min(255, 5 * diff[i][j])
                    else:
                        diffImg[i][j][:] = srcImg[i][j]

            fileName = ROOT + '/clustering/diff/{}_{}/img{}.png'.format(srcClassNum, compClassNum, idx)
            cv2.imwrite(fileName, diffImg)
            fileNames.append(fileName)

        # クラスタリングする
        fileName = ROOT + '/clustering/diff/{}_{}/ani.gif'.format(srcClassNum, compClassNum)
        mylib.image.makeGifAnime(fileNames, 3, fileName)


