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

# フォルダの作成
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

ROOT = os.getcwd()
VIEW_LIST = ['view0,0', 'view90,0', 'view90,90', 'view45,45']
for dirName in glob.glob('part*'):
    print dirName
    # 学習エラーをグラフに表示する
    print '- draw error.'
    arg = "path='{}/{}'; limit={}".format(ROOT, dirName, 10000)
    exeName = 'gnuplot/error.gp'
    doGnuplot(arg, exeName)

    # 中間層のニューロンの出力をグラフに表示する
    print '- draw output of middle layer.'
    mkdir('{}/{}/outNeuron'.format(ROOT, dirName))
    # trainディレクトリの記録
    print '  - train directory'
    mkdir('{}/{}/outNeuron/train'.format(ROOT, dirName))
    for viewDir in VIEW_LIST:
        mkdir('{}/{}/outNeuron/train/{}'.format(ROOT, dirName, viewDir))
    arg = "path='{}/{}'; subDirName='train'; limit={}".format(ROOT, dirName, 10000)
    exeName = 'gnuplot/3DNeuron.gp'
    doGnuplot(arg, exeName)
    # testディレクトリの記録
    mkdir('{}/{}/outNeuron/test'.format(ROOT, dirName))
    for swingNum in range(100):
        mkdir('{}/{}/outNeuron/test/{}'.format(ROOT, dirName, swingNum))
        for viewDir in VIEW_LIST:
            mkdir('{}/{}/outNeuron/test/{}/{}'.format(ROOT, dirName, swingNum, viewDir))
    print '  - test directory'
    arg = "path='{}/{}'; subDirName='test'; limit={}; swingLimit={}".format(ROOT, dirName, 10000, 100)
    exeName = 'gnuplot/3DNeuron_swing.gp'
    doGnuplot(arg, exeName)
