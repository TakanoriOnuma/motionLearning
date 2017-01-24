# -*- coding: utf-8 -*-
import os
import glob

# 自作ライブラリのパスを設定してから読み込む
import sys
sys.path.append('C:\Python27\motionLearning')
import mylib

ROOT = os.getcwd()
for dirName in glob.glob('second/part*'):
    print dirName
    reporter = mylib.log.Reporter(ROOT + '/gnuplot', ROOT)
    reporter.setReportDirName(ROOT + '/' + dirName)

    # 記録
    #reporter.drawError()
    reporter.calcFinalError()
    #reporter.integrateTestSwing()
    #reporter.drawOutputInMiddleLayer()
    #reporter.saveOutputOfDirectActivation()
    #reporter.saveIdentityMapping()
    #reporter.normalizeSwingPoints(30)
    #reporter.saveClustering(10)
    #reporter.drawDistribution()
