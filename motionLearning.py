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
def createDir():
    if not os.path.exists('output'):
        os.mkdir('output')

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
def createInputDataList(mnistNumList, trainDataNum):
    inpList = []
    for num in mnistNumList:
        for i in range(trainDataNum):
            fileName = 'MNIST/{0}/{1}/mnist{2}.png'.format('train', num, i)
            img = cv2.imread(fileName)
            inpList.append(makeInputData(img))
    return inpList

# 画像から入力データを作成する
def makeInputData(img):
    data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

MNIST_NUM_LIST = [0, 1, 4]
createDir()

gpuFlag = False

# input and output vector
x_train = np.array(createInputDataList(MNIST_NUM_LIST, 100)).astype(np.float32)
N = len(x_train)

# get image size
IMG_SIZE   = len(x_train[0])
IMG_HEIGHT = int(math.sqrt(IMG_SIZE))
IMG_WIDTH  = IMG_HEIGHT

# model definition
model = MyChain(IMG_SIZE, 100, 30, 2, 30, 100, IMG_SIZE, bias=True)

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
    sum_loss = 0
    perm = np.random.permutation(N)
    for i in range(0, N, batchsize):
        model.zerograds()
        optimizer.zero_grads()

        # extract input and output
        if gpuFlag:
            x = Variable(cuda.cupy.asarray(x_train[perm[i:i + batchsize]]))
            t = Variable(cuda.cupy.asarray(x_train[perm[i:i + batchsize]]))
        else:
            x = Variable(x_train[perm[i:i + batchsize]])
            t = Variable(x_train[perm[i:i + batchsize]])
        
        # estimation by model
        y = model(x)

        # save output image
        if isRoundNumber(epoch):
            if not os.path.exists('output/{0}'.format(epoch)):
                os.mkdir('output/{0}'.format(epoch))
            for j in range(0, batchsize):
                if gpuFlag:
                    img = cuda.to_cpu(makeOutputData(y.data[j], IMG_HEIGHT, IMG_WIDTH))
                else:
                    img = makeOutputData(y.data[j], IMG_HEIGHT, IMG_WIDTH)
                cv2.imwrite('output/{0}/mnist{1}.png'.format(epoch, perm[i + j]), img)

        # error correction
        loss = F.mean_squared_error(y, t)
        sum_loss += loss * batchsize

        # feedback and learning
        loss.backward()
        optimizer.update()

    print "{0}: {1}".format(epoch + 1, 0.5 * sum_loss.data / N)
    
