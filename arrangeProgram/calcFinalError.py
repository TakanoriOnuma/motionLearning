# -*- coding: utf-8 -*-

import sys, os, cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import chainer.functions as F
from chainer import Variable, serializers

sys.path.append('C:\Python27\motionLearning')
import mylib

# 実行ディレクトリを設定する
ROOT = sys.argv[1] if len(sys.argv) == 2 else 'part5'

# モデルの読み込み
prop  = json.load(open(ROOT + '/property.json', 'r'))
model = mylib.NN.MyChain(*prop['nums'], bias=False)
serializers.load_npz(ROOT + '/my.model', model)

imgDir  = 'C:/Python27/motionLearning/learning/IMAGES'
imgDir += '/{}/{}/{}'.format(prop['IMG_DIR'], prop['DATA_TYPE'], 'swing')

labels = ['train', 'test']
dataLists = mylib.image.readDataList(imgDir, prop['SWING_NUM'], prop['TRAIN_SWING'])

fError = open(ROOT + '/finalError.dat', 'w')
for i in range(len(labels)):
    x = Variable(dataLists[i])
    t = Variable(x.data)
    y = model(x)
    loss = F.mean_squared_error(y, t)
    fError.write('{}\t{}\n'.format(labels[i], 0.5 * loss.data))
fError.close()
