# -*- coding: utf-8 -*-
import sys, os, cv2
import numpy
import json
import glob
from chainer import Variable, serializers

sys.path.append('C:\Python27\motionLearning')
import mylib

# 実行ディレクトリを設定する
ROOT = sys.argv[1] if len(sys.argv) == 3 else 'C:/Python27/motionLearning/arrangeProgram/part1'
clusterNum = int(sys.argv[2]) if len(sys.argv) == 3 else 3

# モデルの読み込み
prop  = json.load(open(ROOT + '/property.json', 'r'))
model = mylib.NN.MyChain(*prop['nums'], bias=False)
serializers.load_npz(ROOT + '/my.model', model)

# アニメーション画像から軌跡を作成する
for classNum in range(clusterNum):
    normDir = ROOT + '/clustering/{}/norm'.format(classNum)

    num = len(glob.glob(normDir + '/img*.png'))
    imgs = []
    for i in range(num):
        img = cv2.imread(normDir + '/img{}.png'.format(i))
        imgs.append(mylib.image.makeInputData(img, 'data', 0))
    imgs = numpy.array(imgs).astype(numpy.float32)

    x = Variable(imgs)
    y = model(x)
    mid = model.getMiddleValue()

    mylib.point.savePoints(normDir + '/norm.dat', mid.data)
    mylib.point.drawPoints(normDir, 'norm', 'out_neuron{}_swing/norm{}'.format(prop['TRAIN_NUM'], classNum))
