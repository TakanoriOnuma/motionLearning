#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, FunctionSet, optimizers, cuda, serializers
from chainer import Link, Chain, ChainList

import math
import random
from PIL import Image
import os
import cv2

# ディレクトリの作成
def createDir(dirNames, trainTypes):
    for dirName in dirNames:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        for trainType in trainTypes:
            if not os.path.exists('{0}/{1}'.format(dirName, trainType)):
                os.mkdir('{0}/{1}'.format(dirName, trainType))

# キリのいい数値か調べる
def isRoundNumber(num):
    if num == 0:
        return True
    digits = []
    while num > 0:
        digits.append(num % 10)
        num /= 10
    digits = digits[:-1]
    return all(digit == 0 for digit in digits)

# 丁度対数スケールになっているか調べる
def isJustLogScale(num):
    if num < 10:
        return False
    log = math.log10(num)
    return (log - math.ceil(log)) == 0.0

# 入力データリストを作成する
def createInputDataList(dataType, trainType, trainDataNum):
    inpList = []
    emphaList = []
    for i in range(1, trainDataNum):
        fileName = 'images/{0}/{1}/ren{2}.jpg'.format(dataType, trainType, str(i).zfill(6))
        #fileName = 'images/{0}/{1}/diff/img{2}.jpg'.format(dataType, trainType, str(i + 1))
        img = cv2.imread(fileName)
        inpList.append(makeInputData(img, 'data'))

        fileName = 'images/{0}/{1}/mask/img{2}.jpg'.format(dataType, trainType, i)
        img = cv2.imread(fileName)
        emphaList.append(makeInputData(img, 'empha'))
    return inpList, emphaList

# 画像から入力データを作成する
# inpType: input, empha
def makeInputData(img, inpType):
    data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = cv2.resize(data, None, fx=0.2, fy=0.2)
    data = data.reshape(data.shape[0] * data.shape[1]).astype('float32')
    if inpType == 'data':
        #data = 0.8 * (data / 255.0) + 0.1
        data = data / 318.75 + 0.1
    elif inpType == 'empha':
        data = (EMPHA_VALUE - 1) * (data / 255) + 1
    return data    

# 出力データから画像を作成する
def makeOutputData(out, height, width):
    #img = 255 * (out - 0.1) / 0.8
    img = 318.75 * (out - 0.1)
    img = np.clip(img, 0.0, 255.0)
    img = img.astype('uint8').reshape((height, width))
    return img

# 学習状況を記録する
def writeTrainingProperty(fileName):
    f = open(fileName, 'w')
    f.write('NN:' + '-'.join([str(x) for x in model.layers]) + '\n')
    f.write('BETA:' + str(model.beta) + '\n')
    f.write('DATA_NUM:' + str(DATA_NUM) + '\n')
    f.write('TRAIN_NUM:' + str(TRAIN_NUM) + '\n')
    f.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
    if isinstance(optimizer, optimizers.MomentumSGD):
        f.write('LEARNING_RATE:' + str(optimizer.lr) + '\n')
        f.write('MOMENTUM:' + str(optimizer.momentum) + '\n')
        f.write('LR_DECAY:' + str(LR_DECAY) + '\n')

class MyChain(ChainList):
    def __init__(self, *layers, **options):
        super(MyChain, self).__init__()
        opt = {
            'beta' : 0.5,
            'bias' : False,
            'geneFuncW' : np.random.randn
        }
        for key in options:
            if key not in opt.keys():
                print 'undefined key: {0}'.format(key)
            opt[key] = options[key]

        self.beta = opt['beta']
        self.layers = layers
        for i in range(len(layers) - 1):
            initW = opt['geneFuncW'](layers[i + 1], layers[i])
            self.add_link(L.Linear(layers[i], layers[i + 1], nobias=(not opt['bias']), initialW=initW))

    def __call__(self, x, train=False):
        self.value = [None] * (len(self) + 1)
        self.value[0] = x
        for i in range(len(self)):
            self.value[i + 1] = F.sigmoid(self.beta * self[i](self.value[i]))
        return self.value[-1]

# set properties
gpuFlag  = True
dataType = 'normal'
EMPHA_VALUE = 2
DATA_NUM = 101
TRAIN_NUM = 100000
BATCH_SIZE = 1
optimizer = optimizers.MomentumSGD(1.0, 0.8)
LR_DECAY = 1.0

dirNames   = ['output', 'middle']
trainTypes = ['train', 'test']

createDir(dirNames, trainTypes)

# input and output vector
inpDataList = {}
inpDataList['data'] = {}
inpDataList['empha'] = {}
for trainType in trainTypes:
    dataList, emphaList = createInputDataList(dataType, trainType, DATA_NUM)
    inpDataList['data'][trainType] = np.array(dataList).astype(np.float32)
    inpDataList['empha'][trainType] = np.array(emphaList).astype(np.float32)

# get image size
IMG_SIZE   = len(inpDataList['data']['train'][0])
IMG_HEIGHT = int(math.sqrt(IMG_SIZE))
IMG_WIDTH  = IMG_HEIGHT

# model definition
model = MyChain(IMG_SIZE, 100, 30, 3, 30, 100, IMG_SIZE, bias=False)
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
    if isRoundNumber(epoch):
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
                    img = cuda.to_cpu(makeOutputData(y.data[i], IMG_HEIGHT, IMG_WIDTH))
                else:
                    img = makeOutputData(y.data[i], IMG_HEIGHT, IMG_WIDTH)
                cv2.imwrite('output/{0}/{1}/img{2}.jpg'.format(trainType, epoch, i + 1), img)

                # save middle data
                fMiddle.write('\t'.join([str(value) for value in model.value[MIDDLE_LAYER_NUM].data[i]]) + '\n')
            fMiddle.close()
                
        print ""
        fError.write('\n')

    # adjust learning rate
    if isJustLogScale(epoch):
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
    
