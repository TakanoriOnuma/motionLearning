#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
from chainer import Variable, optimizers, cuda, serializers

import math
import random
import os
import cv2

import sys
ROOT = 'C:/Python27/motionLearning/'
sys.path.append(ROOT)
import mylib

# ディレクトリの作成
def createDir(dirNames, trainTypes):
    for dirName in dirNames:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        for trainType in trainTypes:
            if not os.path.exists('{0}/{1}'.format(dirName, trainType)):
                os.mkdir('{0}/{1}'.format(dirName, trainType))
        if not os.path.exists('{}/{}'.format(dirName, 'test')):
            os.mkdir('{0}/{1}'.format(dirName, 'test'))
        for swingNum in range(PROP['SWING_NUM']):
            if not os.path.exists('{}/{}/{}'.format(dirName, 'test', swingNum)):
                os.mkdir('{}/{}/{}'.format(dirName, 'test', swingNum))

# set properties
PROP = {
    'GPU_FLAG'    : True,
    'DATA_TYPE'   : 'normal',
    'IMG_DIR'     : 'images5',
    'EMPHA_VALUE' : 1,
    'SWING_NUM'   : 100,
    'DATA_NUM'    : 560,
    'TRAIN_NUM'   : 10000,
    'BATCH_SIZE'  : 1
}

# set optimizer
optimizer = optimizers.MomentumSGD(1.0, 0.8)
if isinstance(optimizer, optimizers.MomentumSGD):
    PROP['LEARNING_RATE'] = optimizer.lr
    PROP['MOMENTUM']      = optimizer.momentum
    PROP['LR_DECAY']      = 1.0

# createDirs
dirNames   = ['output', 'middle']
trainTypes = ['train']
createDir(dirNames, trainTypes)

# input and output vector
inpDataList = {}
inpDataList['data'] = {}
inpDataList['empha'] = {}
for trainType in trainTypes:
    rootDir = '../IMAGES/{}/{}/{}'.format(PROP['IMG_DIR'], PROP['DATA_TYPE'], trainType)
    dataList, emphaList = mylib.image.createInputDataList(rootDir, PROP['DATA_NUM'], PROP['EMPHA_VALUE'])
    inpDataList['data'][trainType]  = np.array(dataList).astype(np.float32)
    inpDataList['empha'][trainType] = np.array(emphaList).astype(np.float32)

testDataList = []
for i in range(PROP['SWING_NUM']):
    rootDir = '../IMAGES/{}/{}/{}/{}'.format(PROP['IMG_DIR'], PROP['DATA_TYPE'], 'test', i)
    dataList = mylib.image.createInputDataList2(rootDir)
    testDataList.append(np.array(dataList).astype(np.float32))

# get image size
img = cv2.imread(rootDir + '/img{}.png'.format(0))
PROP['IMG_SIZE']   = int(img.shape[0] * img.shape[1])
PROP['IMG_HEIGHT'] = int(img.shape[0])
PROP['IMG_WIDTH']  = int(img.shape[1])

# model definition
model = mylib.NN.MyChain(PROP['IMG_SIZE'], 100, 30, 3, 30, 100, PROP['IMG_SIZE'], bias=False)
PROP['NN']          = '-'.join([str(x) for x in model.layers])
PROP['nums']        = [x for x in model.layers]
PROP['midLayerNum'] = len(model) / 2

# set gpu mode if GPU_FLAG is true
if PROP['GPU_FLAG']:
    model.to_gpu()
optimizer.setup(model)

mylib.property.writeProperty('property.txt', PROP)
fError = open('error.dat', 'w')
fError.write('# epoch\t' + '\t'.join(trainTypes) + '\n')
# main routine
for epoch in range(0, PROP['TRAIN_NUM'] + 1):
    # write log
    if mylib.util.isRoundNumber(epoch):
        print "{0}:".format(epoch),
        fError.write(str(epoch))
        for trainType in trainTypes:
            # set input and target values
            empha = mylib.NN.cupyArray(inpDataList['empha'][trainType], PROP['GPU_FLAG'])
            x     = Variable(mylib.NN.cupyArray(inpDataList['data'][trainType], PROP['GPU_FLAG']))
            t     = Variable(empha * x.data)

            y = model(x)
            loss = F.mean_squared_error(empha * y, t)
            print 0.5 * loss.data,
            fError.write('\t' + str(0.5 * loss.data))

            if not os.path.exists('output/{0}/{1}'.format(trainType, epoch)):
                os.mkdir('output/{0}/{1}'.format(trainType, epoch))

            fMiddle = open('middle/{0}/middle{1}.dat'.format(trainType, epoch), 'w')
            fMiddle.write('# ')
            fMiddle.write('\t'.join(['middle{0}'.format(i) for i in range(PROP['nums'][PROP['midLayerNum']])]) + '\n')
            for i in range(0, PROP['DATA_NUM']):
                # save output image
                img = mylib.image.makeOutputData(y.data[i], PROP['IMG_HEIGHT'], PROP['IMG_WIDTH'])
                if PROP['GPU_FLAG']:
                    img = cuda.to_cpu(img)                
                cv2.imwrite('output/{0}/{1}/img{2}.png'.format(trainType, epoch, i + 1), img)

                # save middle data
                fMiddle.write('\t'.join([str(value) for value in model.value[PROP['midLayerNum']].data[i]]) + '\n')
            fMiddle.close()
                
        print ""
        fError.write('\n')

        # テストデータの入力
        for swingNum in range(PROP['SWING_NUM']):
            x = Variable(mylib.NN.cupyArray(testDataList[swingNum], PROP['GPU_FLAG']))
            y = model(x)
            if not os.path.exists('output/test/{}/{}'.format(swingNum, epoch)):
                os.mkdir('output/test/{}/{}'.format(swingNum, epoch))

            fMiddle = open('middle/test/{}/middle{}.dat'.format(swingNum, epoch), 'w')
            fMiddle.write('# ')
            fMiddle.write('\t'.join(['middle{0}'.format(i) for i in range(PROP['nums'][PROP['midLayerNum']])]) + '\n')
            for i in range(len(testDataList[swingNum])):
                # save middle data
                fMiddle.write('\t'.join([str(value) for value in model.value[PROP['midLayerNum']].data[i]]) + '\n')
            fMiddle.close()               

        # 最後の学習まで行っていたら終了する
        if epoch == PROP['TRAIN_NUM']:
            break
                  
    # adjust learning rate
    if mylib.util.isJustLogScale(epoch):
        optimizer.lr *= PROP['LR_DECAY']
        print "LR_DECAY:", str(optimizer.lr)

    # learning
    perm = np.random.permutation(PROP['DATA_NUM'])
    for i in range(0, PROP['DATA_NUM'], PROP['BATCH_SIZE']):
        model.zerograds()
        optimizer.zero_grads()

        # extract input and target values
        extraRange = perm[i:i + PROP['BATCH_SIZE']]
        empha = mylib.NN.cupyArray(inpDataList['empha']['train'][extraRange], PROP['GPU_FLAG'])
        x     = Variable(mylib.NN.cupyArray(inpDataList['data']['train'][extraRange], PROP['GPU_FLAG']))
        t     = Variable(empha * x.data)
        
        # estimation by model
        y = model(x, train=True)

        # error correction
        loss = F.mean_squared_error(empha * y, t)

        # feedback and learning
        loss.backward()
        optimizer.update()

fError.close()
serializers.save_npz('my.model', model)
    
