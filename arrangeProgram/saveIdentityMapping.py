# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
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

for swingNum in range(prop['SWING_NUM']):
    x = Variable(mylib.NN.cupyArray(swingDataList[swingNum], False))
    y = model(x)

    mylib.util.mkdir('{}/output/swing'.format(DIR_NAME))

    for i in range(len(swingDataList[swingNum])):
        mylib.util.mkdir('{}/output/swing/swing{}'.format(DIR_NAME, i))
        img = mylib.image.makeOutputData(y.data[i], prop['IMG_HEIGHT'], prop['IMG_WIDTH'])
        cv2.imwrite('{}/output/swing/swing{}/img{}.png'.format(DIR_NAME, swingNum, i), img)
