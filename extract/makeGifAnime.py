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
    for dirName in glob.glob('{}/{}/test/[0-9]*'.format(ROOT, dataType)):
        print dirName
        imgFiles = glob.glob('{}/img*.png'.format(dirName))
        numberingFiles = ['{}/img{}.png'.format(dirName, i) for i in range(len(imgFiles))]
        mylib.image.makeGifAnime(numberingFiles, 3, '{}/ani.gif'.format(dirName))

# gifアニメをまとめてみるHTMLを作成
for dataType in dataTypes:
    dirName = '{}/{}/test'.format(ROOT, dataType)
    fHtml = open('{}/anigif.html'.format(dirName), 'w')
    fHtml.write('<table border="1">\n')
    i = 0
    while os.path.exists('{}/{}/ani.gif'.format(dirName, i)):
        if i % 10 == 0:
            fHtml.write('  <tr align="center">\n')
            fHtml.write('    ')
        fHtml.write('<td><img src="{0}/ani.gif"><br>{0}</td>'.format(i))
            
        if i % 10 == 9:
            fHtml.write('\n')
            fHtml.write('  </tr>\n')
        i += 1
    fHtml.write('</table>')
    fHtml.close()
    
