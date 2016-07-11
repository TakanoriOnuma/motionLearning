#!/usr/bin/env python
# coding:utf-8

import numpy as np
import cv2

EMPHA_VALUE = 1

# 入力データリストを作成する
def createInputDataList(rootDir, dataType, trainType, trainDataNum):
    inpList = []
    emphaList = []
    for i in range(1, trainDataNum + 1):
        fileName = '{0}/{1}/{2}/img{3}.png'.format(rootDir, dataType, trainType, i + 1)
        img = cv2.imread(fileName)
        inpList.append(makeInputData(img, 'data'))

        fileName = '{0}/{1}/{2}/mask/img{3}.png'.format(rootDir, dataType, trainType, i)
        img = cv2.imread(fileName)
        emphaList.append(makeInputData(img, 'empha'))
    return inpList, emphaList

# 画像から入力データを作成する
# inpType: input, empha
def makeInputData(img, inpType):
    data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
