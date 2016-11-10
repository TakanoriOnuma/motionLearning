# -*- coding: utf-8 -*-
import os
import glob

import sys
sys.path.append('C:/Python27/motionLearning/')
import mylib

ROOT = 'images5'
dataTypes = ['normal']

# gifアニメを作成する
for dataType in dataTypes:
    maxNum = 0
    for dirName in glob.glob('{}/{}/swing/[0-9]*'.format(ROOT, dataType)):
        num = len(glob.glob('{}/img*.png'.format(dirName)))
        maxNum = num if num > maxNum else maxNum
    print maxNum
    for dirName in glob.glob('{}/{}/swing/[0-9]*'.format(ROOT, dataType)):
        print dirName
        num = len(glob.glob('{}/img*.png'.format(dirName)))
        numberingFiles = ['{}/img{}.png'.format(dirName, i) for i in range(num)]
        mylib.image.makeGifAnime(numberingFiles, 3, '{}/ani.gif'.format(dirName))
        # 回数を合わせるため後ろの画像を追加する
        for i in range(maxNum - num):
            numberingFiles.append('{}/img{}.png'.format(dirName, num - 1))
        mylib.image.makeGifAnime(numberingFiles, 3, '{}/ani2.gif'.format(dirName))

# gifアニメをまとめてみるHTMLを作成
for dataType in dataTypes:
    dirName = '{}/{}/swing'.format(ROOT, dataType)
    fHtml = open('{}/anigif.html'.format(dirName), 'w')
    fHtml.write('<table border="1">\n')
    i = 0
    while os.path.exists('{}/{}/ani2.gif'.format(dirName, i)):
        if i % 10 == 0:
            fHtml.write('  <tr align="center">\n')
            fHtml.write('    ')
        fHtml.write('<td><img src="{0}/ani2.gif"><br>{0}</td>'.format(i))
            
        if i % 10 == 9:
            fHtml.write('\n')
            fHtml.write('  </tr>\n')
        i += 1
    fHtml.write('</table>')
    fHtml.close()
    
