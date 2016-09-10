# -*- coding: utf-8 -*-
import subprocess

dirName = 'C:/Python27/motionLearning/arrangeProgram/part1'

arg = "path='{}';limit={}".format(dirName, 10000)
exeName = 'gnuplot/error.gp'
p = subprocess.Popen(['gnuplot', '-e', arg, exeName], stderr=subprocess.PIPE)
for line in p.stderr:
    print line,
