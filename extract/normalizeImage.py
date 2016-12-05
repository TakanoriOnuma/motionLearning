# -*- coding: utf-8 -*-
import glob
import re
import os
import cv2
import numpy

# 自作ライブラリのパスを設定してから読み込む
import sys
sys.path.append('C:\Python27\motionLearning')
import mylib.util

ROOT = 'images5/normal'

SWING_NUM = 100

for swingNum in range(SWING_NUM):
    # swingフォルダを見る
    searchFileName = '{}/swing/{}/*.png'.format(ROOT, swingNum)
    num = len(glob.glob(searchFileName))

    # 画像を1次元データとして読み込む
    imgs = []
    for idx in range(num):
        img = cv2.imread('{}/swing/{}/img{}.png'.format(ROOT, swingNum, idx))
        height = img.shape[0]
        width  = img.shape[1]
        imgs.append(mylib.image.makeInputData(img, 'data', 0))

    # 30点で正規化する
    normImgs = numpy.array(mylib.point.normalizePoints(imgs, 30)).astype(numpy.float32)

    # 画像に保存する
    normDir = '{}/swing/{}/norm'.format(ROOT, swingNum, idx)
    mylib.util.mkdir(normDir)
    normImgNames = []
    for idx in range(len(normImgs)):
        img = mylib.image.makeOutputData(normImgs[idx], height, width)
        imgFileName = normDir + '/img{}.png'.format(idx)
        cv2.imwrite(imgFileName, img)
        normImgNames.append(imgFileName)

    # アニメーション化する
    mylib.image.makeGifAnime(normImgNames, 3, normDir + '/ani.gif')
