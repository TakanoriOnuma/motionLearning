# -*- coding: utf-8 -*-
import glob

import sys
sys.path.append('C:/Python27/motionLearning/')
import mylib

ROOT = 'images5'
dataTypes = ['normal']

for dataType in dataTypes:
    for dirName in glob.glob('{}/{}/test/[0-9]*'.format(ROOT, dataType)):
        print dirName
        imgFiles = glob.glob('{}/img*.png'.format(dirName))
        numberingFiles = ['{}/img{}.png'.format(dirName, i) for i in range(len(imgFiles))]
        mylib.image.makeGifAnime(numberingFiles, 3, '{}/ani.gif'.format(dirName))
