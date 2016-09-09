# -*- coding: utf-8 -*-

import os
import cv2
import numpy

dirName = 'images5/normal/train/'

# 差分画像を保存する
if not os.path.exists(dirName + '/diff'):
    os.mkdir(dirName + '/diff')

i = 0
fileName = '{}/img{}.png'.format(dirName, i)
prevImg  = cv2.imread(fileName).astype(int)
i += 1
fileName = '{}/img{}.png'.format(dirName, i)
while os.path.exists(fileName):
    img      = cv2.imread(fileName).astype(int)
    diffImg  = abs(img - prevImg).astype(numpy.uint8)
    saveName = '{}/diff/img{}.png'.format(dirName, i - 1)
    cv2.imwrite(saveName, diffImg)
    prevImg = img
    i += 1
    fileName = '{}/img{}.png'.format(dirName, i)

# マスク画像を作成する
if not os.path.exists(dirName + '/mask'):
    os.mkdir(dirName + '/mask')

THRETHOLD = 20
i = 0
fileName     = '{}/diff/img{}.png'.format(dirName, i)
prevImg      = cv2.imread(fileName)
ret, prevImg = cv2.threshold(prevImg, THRETHOLD, 1, cv2.THRESH_BINARY)
i += 1
fileName = '{}/diff/img{}.png'.format(dirName, i)
while os.path.exists(fileName):
    img      = cv2.imread(fileName)
    ret, img = cv2.threshold(img, THRETHOLD, 1, cv2.THRESH_BINARY)
    dotImg   = (prevImg * img) * 255
    saveName = '{}/mask/img{}.png'.format(dirName, i)
    cv2.imwrite(saveName, dotImg)
    prevImg = img
    i += 1
    fileName = '{}/diff/img{}.png'.format(dirName, i)

