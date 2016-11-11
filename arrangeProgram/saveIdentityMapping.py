# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import chainer.functions as F
from chainer import Variable, serializers

import sys
ROOT = 'C:\Python27\motionLearning'
sys.path.append(ROOT)
import mylib

DIR_NAME = sys.argv[1] if len(sys.argv) == 2 else 'test'
prop = json.load(open(DIR_NAME + '/property.json', 'r'))
model = mylib.NN.MyChain(*prop['nums'], bias=False)
serializers.load_npz(DIR_NAME + '/my.model', model)

swingDataList = []
for i in range(prop['SWING_NUM']):
    rootDir = '{}/learning/IMAGES/{}/{}/{}/{}' \
              .format(ROOT, prop['IMG_DIR'], prop['DATA_TYPE'], 'swing', i)
    dataList = mylib.image.createInputSwingDataList(rootDir)
    swingDataList.append(np.array(dataList).astype(np.float32))

errors = []
for swingNum in range(prop['SWING_NUM']):
    x = Variable(mylib.NN.cupyArray(swingDataList[swingNum], False))
    t = Variable(x.data)
    
    y = model(x)
    loss = F.mean_squared_error(y, t)
    errors.append(loss.data)

    mylib.util.mkdir('{}/output/swing'.format(DIR_NAME))
    for i in range(len(swingDataList[swingNum])):
        mylib.util.mkdir('{}/output/swing/swing{}'.format(DIR_NAME, i))
        img = mylib.image.makeOutputData(y.data[i], prop['IMG_HEIGHT'], prop['IMG_WIDTH'])
        cv2.imwrite('{}/output/swing/swing{}/img{}.png'.format(DIR_NAME, swingNum, i), img)

# エラーを記録
errors = np.array(errors)
fError = open(DIR_NAME + '/output/swing/error.dat', 'w')
fError.write('# swingNum\t' + 'error\n')
for swingNum in range(prop['SWING_NUM']):
    fError.write('\t'.join([str(swingNum), str(errors[swingNum])]) + '\n')
fError.close()

# 最大値を1にした相対エラーを記録
errors = errors / errors.max()
fError = open(DIR_NAME + '/output/swing/error_prop.dat', 'w')
fError.write('# swingNum\t' + 'error\n')
for swingNum in range(prop['SWING_NUM']):
    fError.write('\t'.join([str(swingNum), str(errors[swingNum])]) + '\n')
fError.close()
