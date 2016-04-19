#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, FunctionSet, optimizers, cuda
from chainer import Link, Chain, ChainList

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
            img = Image.open(fileName)
            inpList.append(makeInputData(img))
    return inpList

# 画像から入力データを作成する
def makeInputData(img):
    inpData = []
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            value = img.getpixel((x, y))[0] / 255.0
            value = 0.8 * value + 0.1
            inpData.append(value)
    return inpData

# 出力データから画像を作成する
def makeOutputData(out):
    img = Image.new('RGB', (28, 28))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            value = out[y * img.size[0] + x] - 0.1
            value = value if value > 0.0 else 0.0
            value = (value / 0.8) * 255
            value = int(value)
            img.putpixel((x, y), (value, value, value))
    return img

# 高速化を考慮して、出力データから画像を作成する
def makeOutputData2(out):
    #img = 255 * (out - 0.1) / 0.8
    img = 318.75 * (out - 0.1)
    img = np.clip(img, 0.0, 255.0)
    img = img.astype('uint8').reshape((28, 28))
    return img

class MyChain(ChainList):
    def __init__(self):
        super(MyChain, self).__init__(
            L.Linear(784, 100, nobias=True, initialW=np.random.randn(100, 784)),
            L.Linear(100, 30, nobias=True, initialW=np.random.randn(30, 100)),
            L.Linear(30, 2, nobias=True, initialW=np.random.randn(2, 30)),
            L.Linear(2, 30, nobias=True, initialW=np.random.randn(30, 2)),
            L.Linear(30, 100, nobias=True, initialW=np.random.randn(100, 30)),
            L.Linear(100, 784, nobias=True, initialW=np.random.randn(784, 100))
        )

    def __call__(self, x):
        self.value = [None] * (len(self) + 1)
        self.value[0] = x
        for i in range(len(self)):
            self.value[i + 1] = F.sigmoid(self[i](self.value[i]))
        return self.value[-1]

MNIST_NUM_LIST = [0, 1, 4]
createDir()

gpuFlag = True

# model definition
model = MyChain()
if gpuFlag:
    model.to_gpu()
optimizer = optimizers.MomentumSGD(5.0, 0.8)
#optimizer = optimizers.Adam()
optimizer.setup(model)

# number of learning
times = 500

# input and output vector
x_train = np.array(createInputDataList(MNIST_NUM_LIST, 100)).astype(np.float32)
N = len(x_train)

# main routine
batchsize = 1
for epoch in range(0, times):
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
                    img = cuda.to_cpu(makeOutputData2(y.data[j]))
                else:
                    img = makeOutputData2(y.data[j])
                cv2.imwrite('output/{0}/mnist{1}.png'.format(epoch, perm[i + j]), img)

        # error correction
        loss = F.mean_squared_error(y, t)
        sum_loss += loss * batchsize

        # feedback and learning
        loss.backward()
        optimizer.update()

    print "{0}: {1}".format(epoch + 1, 0.5 * sum_loss.data / N)
    

