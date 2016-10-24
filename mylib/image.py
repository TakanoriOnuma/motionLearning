#!/usr/bin/env python
# coding:utf-8

import numpy as np
import cv2
import os
import subprocess

# 入力データリストを作成する
def createInputDataList(rootDir, trainDataNum, emphaValue):
    inpList = []
    emphaList = []
    for i in range(1, trainDataNum + 1):
        fileName = '{}/img{}.png'.format(rootDir, i)
        img = cv2.imread(fileName)
        inpList.append(makeInputData(img, 'data', emphaValue))

        fileName = '{}/mask/img{}.png'.format(rootDir, i)
        img = cv2.imread(fileName)
        emphaList.append(makeInputData(img, 'empha', emphaValue))
    return inpList, emphaList

# 1スイング分の入力データリストを作成する
def createInputSwingDataList(rootDir):
    inpList = []
    i = 0
    fileName = '{}/img{}.png'.format(rootDir, i)
    while os.path.exists(fileName):
        img = cv2.imread(fileName)
        inpList.append(makeInputData(img, 'data', 0))
        i += 1
        fileName = '{}/img{}.png'.format(rootDir, i)
    return inpList

# 画像から入力データを作成する
# inpType: input, empha
def makeInputData(img, inpType, emphaValue):
    data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = data.reshape(data.shape[0] * data.shape[1]).astype('float32')
    if inpType == 'data':
        #data = 0.8 * (data / 255.0) + 0.1
        data = data / 318.75 + 0.1
    elif inpType == 'empha':
        data = (emphaValue - 1) * (data / 255) + 1
    return data    

# 出力データから画像を作成する
def makeOutputData(out, height, width):
    #img = 255 * (out - 0.1) / 0.8
    img = 318.75 * (out - 0.1)
    img = np.clip(img, 0.0, 255.0)
    img = img.astype('uint8').reshape((height, width))
    return img

# 4枚の画像を2x2として結合する
# img4に何もセットしない場合は白画像をくっつける
def concat(img1, img2, img3, img4 = None):
    if img4 is None:
        img4 = img3.copy()
        img4[:] = 255

    img_row1 = cv2.hconcat([img1, img2])
    img_row2 = cv2.hconcat([img3, img4])
    return cv2.vconcat([img_row1, img_row2])


# 複数画像を1つのgifアニメに変換する
def makeGifAnime(images, delay, fileName):
    cmds = ['convert', '-delay', str(delay), '-loop', str(0)]
    cmds.extend(images)
    cmds.append(fileName)
    p = subprocess.Popen(cmds, stderr=subprocess.PIPE, shell=True)
    # エラーがあったら出力する
    for line in p.stderr:
        print line,
