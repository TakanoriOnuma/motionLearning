#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
from chainer import Variable, optimizers, cuda, serializers

import math
import random
import os
import cv2

import mylib

# ディレクトリの作成
def createDir(dirNames, trainTypes):
    for dirName in dirNames:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        for trainType in trainTypes:
            if not os.path.exists('{0}/{1}'.format(dirName, trainType)):
                os.mkdir('{0}/{1}'.format(dirName, trainType))

# 学習状況を記録する
def writeTrainingProperty(fileName):
    f = open(fileName, 'w')
    f.write('NN:' + '-'.join([str(x) for x in model.layers]) + '\n')
    f.write('BETA:' + str(model.beta) + '\n')
    f.write('EMPHA_VALUE:' + str(EMPHA_VALUE) + '\n')
    f.write('IMG_DIR:' + IMG_DIR + '\n')
    f.write('DATA_NUM:' + str(DATA_NUM) + '\n')
    f.write('TRAIN_NUM:' + str(TRAIN_NUM) + '\n')
    f.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
    if isinstance(optimizer, optimizers.MomentumSGD):
        f.write('LEARNING_RATE:' + str(optimizer.lr) + '\n')
        f.write('MOMENTUM:' + str(optimizer.momentum) + '\n')
        f.write('LR_DECAY:' + str(LR_DECAY) + '\n')

# set properties
gpuFlag  = True
dataType = 'normal'
IMG_DIR  = 'images3'
EMPHA_VALUE = 2
DATA_NUM = 100
TRAIN_NUM = 100000
BATCH_SIZE = 1
optimizer = optimizers.MomentumSGD(1.0, 0.8)
LR_DECAY = 1.0

dirNames   = ['output', 'middle']
trainTypes = ['train', 'normal-test', 'anomaly-test']

createDir(dirNames, trainTypes)

# input and output vector
inpDataList = {}
inpDataList['data'] = {}
inpDataList['empha'] = {}
for trainType in trainTypes:
    dataList, emphaList = mylib.image.createInputDataList(IMG_DIR, dataType, trainType, DATA_NUM)
    inpDataList['data'][trainType] = np.array(dataList).astype(np.float32)
    inpDataList['empha'][trainType] = np.array(emphaList).astype(np.float32)

# get image size
IMG_SIZE   = len(inpDataList['data']['train'][0])
IMG_HEIGHT = int(math.sqrt(IMG_SIZE))
IMG_WIDTH  = IMG_HEIGHT

# model definition
model = mylib.NN.MyChain(IMG_SIZE, 100, 30, 3, 30, 100, IMG_SIZE, bias=False)
MIDDLE_LAYER_NUM = len(model) / 2
MIDDLE_NEURON_NUM = len(model[MIDDLE_LAYER_NUM].W.data[0])

if gpuFlag:
    model.to_gpu()
#optimizer = optimizers.Adam()
optimizer.setup(model)

writeTrainingProperty('property.txt')
fError = open('error.dat', 'w')
fError.write('# epoch\t' + '\t'.join(trainTypes) + '\n')
# main routine
for epoch in range(0, TRAIN_NUM + 1):
    # write log
    if mylib.util.isRoundNumber(epoch):
        print "{0}:".format(epoch),
        fError.write(str(epoch))
        for trainType in trainTypes:
            if gpuFlag:
                empha = cuda.cupy.asarray(inpDataList['empha'][trainType])
                x = Variable(cuda.cupy.asarray(inpDataList['data'][trainType]))
                t = Variable(empha * x.data)
            else:
                empha = inpDataList['empha'][trainType]
                x = Variable(inpDataList['data'][trainType])
                t = Variable(empha * x.data)

            y = model(x)
            loss = F.mean_squared_error(empha * y, t)
            print 0.5 * loss.data,
            fError.write('\t' + str(0.5 * loss.data))

            if not os.path.exists('output/{0}/{1}'.format(trainType, epoch)):
                os.mkdir('output/{0}/{1}'.format(trainType, epoch))

            fMiddle = open('middle/{0}/middle{1}.dat'.format(trainType, epoch), 'w')
            fMiddle.write('# ')
            fMiddle.write('\t'.join(['middle{0}'.format(i) for i in range(MIDDLE_NEURON_NUM)]) + '\n')
            for i in range(0, DATA_NUM - 1):
                # save output image
                if gpuFlag:
                    img = cuda.to_cpu(mylib.image.makeOutputData(y.data[i], IMG_HEIGHT, IMG_WIDTH))
                else:
                    img = mylib.image.makeOutputData(y.data[i], IMG_HEIGHT, IMG_WIDTH)
                cv2.imwrite('output/{0}/{1}/img{2}.png'.format(trainType, epoch, i + 1), img)

                # save middle data
                fMiddle.write('\t'.join([str(value) for value in model.value[MIDDLE_LAYER_NUM].data[i]]) + '\n')
            fMiddle.close()
                
        print ""
        fError.write('\n')

        # 最後の学習まで行っていたら終了する
        if epoch == TRAIN_NUM:
            break
                  
    # adjust learning rate
    if mylib.util.isJustLogScale(epoch):
        optimizer.lr *= LR_DECAY
        print "LR_DECAY:", str(optimizer.lr)

    # learning    
    perm = np.random.permutation(DATA_NUM - 1)
    for i in range(0, DATA_NUM - 1, BATCH_SIZE):
        model.zerograds()
        optimizer.zero_grads()

        # extract input and output
        if gpuFlag:
            empha = cuda.cupy.asarray(inpDataList['empha']['train'][perm[i:i + BATCH_SIZE]])
            x = Variable(cuda.cupy.asarray(inpDataList['data']['train'][perm[i:i + BATCH_SIZE]]))
            t = Variable(empha * x.data)
        else:
            empha = inpDataList['empha']['train'][perm[i:i + BATCH_SIZE]]
            x = Variable(inpDataList['data']['train'][perm[i:i + BATCH_SIZE]])
            t = Variable(empha * x.data)
        
        # estimation by model
        y = model(x, train=True)

        # error correction
        loss = F.mean_squared_error(empha * y, t)

        # feedback and learning
        loss.backward()
        optimizer.update()

fError.close()
serializers.save_npz('my.model', model)
    
