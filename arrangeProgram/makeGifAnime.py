# -*- coding: utf-8 -*-
import os
import glob
import json

import sys
ROOT = 'C:/Python27/motionLearning/'
sys.path.append(ROOT)
import mylib

DIR_NAME = sys.argv[1] if len(sys.argv) == 2 else 'part2'
prop = json.load(open(DIR_NAME + '/property.json', 'r'))

# gifアニメを作成する
maxNum = 0
for dirName in glob.glob('{}/output/swing/swing[0-9]*'.format(DIR_NAME)):
    num = len(glob.glob('{}/img*.png'.format(dirName)))
    maxNum = num if num > maxNum else maxNum
print maxNum
for dirName in glob.glob('{}/output/swing/swing[0-9]*'.format(DIR_NAME)):
    print dirName
    num = len(glob.glob('{}/img*.png'.format(dirName)))
    numberingFiles = ['{}/img{}.png'.format(dirName, i) for i in range(num)]
    mylib.image.makeGifAnime(numberingFiles, 3, '{}/ani.gif'.format(dirName))
    # 回数を合わせるため後ろの画像を追加する
    for i in range(maxNum - num):
        numberingFiles.append('{}/img{}.png'.format(dirName, num - 1))
    mylib.image.makeGifAnime(numberingFiles, 3, '{}/ani2.gif'.format(dirName))
# error量を取得する
fErrors = open('{}/output/swing/error_prop.dat'.format(DIR_NAME), 'r')
errors = []
for line in fErrors:
    if line[0] == '#':
        continue
    swingNum, error = line[:-1].split('\t')
    errors.append(error)

# gifアニメをまとめてみるHTMLを作成
dirName = '{}/output/swing'.format(DIR_NAME)
fHtml = open('{}/anigif.html'.format(dirName), 'w')
fHtml.write('<table border="1">\n')
i = 0
while os.path.exists('{}/swing{}/ani2.gif'.format(dirName, i)):
    if i % 10 == 0:
        fHtml.write('  <tr align="center">\n')
        fHtml.write('    ')
    fHtml.write('<td><img src="swing{0}/ani2.gif"><br>{0}<br>{1}</td>'.format(i, errors[i]))
            
    if i % 10 == 9:
        fHtml.write('\n')
        fHtml.write('  </tr>\n')
    i += 1
fHtml.write('</table>')
fHtml.close()
    
