#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, FunctionSet, optimizers, cuda
from chainer import Link, Chain, ChainList

import math
import random
from PIL import Image
import os
import cv2

# ディレクトリの作成
def createDir(trainTypes):
    if not os.path.exists('output'):
        os.mkdir('output')
    for trainType in trainTypes:
        if not os.path.exists('output/{0}'.format(trainType)):
            os.mkdir('output/{0}'.format(trainType))

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

# 入力データリストを作成する
def createInputDataList(dataType, trainType, trainDataNum):
    inpList = []
    for i in range(trainDataNum):
        fileName = 'images/{0}/{1}/ren{2}.jpg'.format(dataType, trainType, str(i).zfill(6))
        img = cv2.imread(fileName)
        inpList.append(makeInputData(img))
    return inpList

# 画像から入力データを作成する
def makeInputData(img):
    data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = cv2.resize(data, None, fx=0.2, fy=0.2)
    data = data.reshape(data.shape[0] * data.shape[1]).astype('float32')
    #data = 0.8 * (data / 255.0) + 0.1
    data = data / 318.75 + 0.1
    return data    

# 出力データから画像を作成する
def makeOutputData(out, height, width):
    #img = 255 * (out - 0.1) / 0.8
    img = 318.75 * (out - 0.1)
    img = np.clip(img, 0.0, 255.0)
    img = img.astype('uint8').reshape((height, width))
    return img

class MyChain(ChainList):
    def __init__(self, *layers, **options):
        super(MyChain, self).__init__()
        opt = {
            'bias' : False,
            'geneFuncW' : np.random.randn
        }
        for key in options:
            if key not in opt.keys():
                print 'undefined key: {0}'.format(key)
            opt[key] = options[key]
            
        for i in range(len(layers) - 1):
            initW = opt['geneFuncW'](layers[i + 1], layers[i])
            self.add_link(L.Linear(layers[i], layers[i + 1], nobias=(not opt['bias']), initialW=initW))

    def __call__(self, x):
        self.value = [None] * (len(self) + 1)
        self.value[0] = x
        for i in range(len(self)):
            self.value[i + 1] = F.sigmoid(self[i](self.value[i]))
        return self.value[-1]

# set properties
gpuFlag  = True
dataType = 'normal'

trainTypes = ['train', 'test']

createDir(trainTypes)

# input and output vector
N = 100
inpDataList = {}
for trainType in trainTypes:
    dataList = createInputDataList(dataType, trainType, N)
    inpDataList[trainType] = np.array(dataList).astype(np.float32)

# get image size
IMG_SIZE   = len(inpDataList['train'][0])
IMG_HEIGHT = int(math.sqrt(IMG_SIZE))
IMG_WIDTH  = IMG_HEIGHT

# model definition
model = MyChain(IMG_SIZE, 100, 30, 10, 30, 100, IMG_SIZE, bias=True)

if gpuFlag:
    model.to_gpu()
optimizer = optimizers.MomentumSGD(5.0, 0.8)
#optimizer = optimizers.Adam()
optimizer.setup(model)

# number of learning
times = 500

# main routine
batchsize = 1
for epoch in range(0, times + 1):
    # write log
    if isRoundNumber(epoch):
        print "{0}:".format(epoch),
        for trainType in trainTypes:
            if gpuFlag:
                x = Variable(cuda.cupy.asarray(inpDataList[trainType]))
                t = Variable(x.data)
            else:
                x = Variable(inpDataList[trainType])
                t = Variable(x.data)

            y = model(x)
            loss = F.mean_squared_error(y, t)
            print 0.5 * loss.data,

            # save output image
            if not os.path.exists('output/{0}/{1}'.format(trainType, epoch)):
                os.mkdir('output/{0}/{1}'.format(trainType, epoch))
            for i in range(N):
                if gpuFlag:
                    img = cuda.to_cpu(makeOutputData(y.data[i], IMG_HEIGHT, IMG_WIDTH))
                else:
                    img = makeOutputData(y.data[i], IMG_HEIGHT, IMG_WIDTH)
                cv2.imwrite('output/{0}/{1}/img{2}.jpg'.format(trainType, epoch, i), img)
        print ""
        
    sum_loss = 0
    perm = np.random.permutation(N)
    for i in range(0, N, batchsize):
        model.zerograds()
        optimizer.zero_grads()

        # extract input and output
        if gpuFlag:
            x = Variable(cuda.cupy.asarray(inpDataList['train'][perm[i:i + batchsize]]))
            t = Variable(x.data)
        else:
            x = Variable(inpDataList['train'][perm[i:i + batchsize]])
            t = Variable(x.data)
        
        # estimation by model
        y = model(x)

        # error correction
        loss = F.mean_squared_error(y, t)
        sum_loss += loss * batchsize

        # feedback and learning
        loss.backward()
        optimizer.update()
    


    
