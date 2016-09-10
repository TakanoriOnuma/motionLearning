# -*- coding: utf-8 -*-
import os
import glob
import subprocess

# gnuplotの実行
def doGnuplot(arg, exeName):
    p = subprocess.Popen(['gnuplot', '-e', arg, exeName], stderr=subprocess.PIPE)
    # エラーがあったら出力する
    for line in p.stderr:
        print line,    

ROOT = os.getcwd()
for dirName in glob.glob('part*'):
    print dirName
    # 学習エラーをグラフに表示する
    arg = "path='{}/{}';limit={}".format(ROOT, dirName, 10000)
    exeName = 'gnuplot/error.gp'
    doGnuplot(arg, exeName)

