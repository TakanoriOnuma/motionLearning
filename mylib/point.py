# -*- coding: utf-8 -*-

import os
import cv2
import json
import glob
import numpy
import shutil

# 自作ライブラリのパスを設定してから読み込む
import sys
sys.path.append('C:\Python27\motionLearning')
import mylib.util

# 点情報を取得する
def getPoints(fileName, separator='\t'):
    fPoints = open(fileName, 'r')
    points = []
    for line in fPoints:
        if line[0] == '#':
            continue
        pts = line[:-1].split(separator)
        for i in range(len(pts)):
            pts[i] = float(pts[i])
        points.append(pts)
    return points

# 点情報をnumPointで規格化
def normalizePoints(points, numPoint):
    step = float(len(points) - 1) / (numPoint - 1)
    normPoints = []
    for i in range(numPoint):
        pt      = i * step
        intPt   = int(pt)
        floatPt = pt - intPt
        # 小数点がない場合は整数の点をそのまま取ってくる
        if floatPt <= 0.1e-10:
            pts = points[intPt]
        # 小数もある場合は内分点の計算をする
        else:
            pts = []
            for j in range(len(points[intPt])):
                pts.append((1 - floatPt) * points[intPt][j] + floatPt * points[intPt + 1][j])
        normPoints.append(pts)
    return normPoints

# 点情報を保存する
def savePoints(fileName, points, separator='\t'):
    fPoints = open(fileName, 'w')
    fPoints.write('# ' + separator.join(['middle{}'.format(i) for i in range(len(points[0]))]) + '\n')
    for i in range(len(points)):
        fPoints.write(separator.join([str(pt) for pt in points[i]]) + '\n')
    fPoints.close()

# gnuplotを通して軌跡を描画する
# fileNameは拡張子なしで渡す
def drawPoints(path, fileName, titleName):
    pts = getPoints('{}/{}.dat'.format(path, fileName))
    dim = len(pts[0])
    # グラフにする
    arg = "path='{}'; fileName='{}'; titleName='{}'".format(path, fileName, titleName)
    exeName = 'gnuplot/{}DNeuron_oneSwing.gp'.format(dim)
    mylib.util.doGnuplot(arg, exeName)
    
    # 3次元の場合は結合処理をする
    if dim == 3:
        img1 = cv2.imread('{}/{}_view45,45.png'.format(path, fileName))
        img2 = cv2.imread('{}/{}_view0,0.png'.format(path, fileName))
        img3 = cv2.imread('{}/{}_view90,0.png'.format(path, fileName))
        img4 = cv2.imread('{}/{}_view90,90.png'.format(path, fileName))
        concatImg = mylib.image.concat(img1, img2, img3, img4)
        cv2.imwrite('{}/{}.png'.format(path, fileName), concatImg)
