# -*- coding: utf-8 -*-
import sys, os, cv2
import numpy as np
import json
from chainer import Variable, serializers

sys.path.append('C:\Python27\motionLearning')
import mylib

# 実行ディレクトリを設定する
ROOT = sys.argv[1] if len(sys.argv) == 3 else 'part4'
clusterNum = sys.argv[2] if len(sys.argv) == 3 else 10

# モデルの読み込み
prop  = json.load(open(ROOT + '/property.json', 'r'))
model = mylib.NN.MyChain(*prop['nums'], bias=False)
serializers.load_npz(ROOT + '/my.model', model)

# クラスタリングの平均情報を基に特徴層に直接入力した時の出力をアニメーション化する
for classNum in range(clusterNum):
    clusteringDir = ROOT + '/clustering/{}'.format(classNum)

    # 点情報を取得する
    fileName = clusteringDir + '/mean.dat'
    pts = np.array(mylib.point.getPoints(fileName)).astype(np.float32)

    # 特徴層に直接入力する
    x = Variable(pts)
    y = model.directActivate(prop['midLayerNum'], x)

    # 画像化する
    mylib.util.mkdir(clusteringDir + '/mean')
    imgFiles = []
    for i in range(len(y.data)):
        out = y.data[i]
        img = mylib.image.makeOutputData(out, prop['IMG_HEIGHT'], prop['IMG_WIDTH'])
        fileName = clusteringDir + '/mean/img{}.png'.format(i)
        cv2.imwrite(fileName, img)
        imgFiles.append(fileName)

    # アニメーションにする
    mylib.image.makeGifAnime(imgFiles, 3, clusteringDir + '/mean/ani.gif')

