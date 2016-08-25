# -*- coding: utf-8 -*-
import sys, os, cv2
import numpy as np
from chainer import Variable, serializers

ROOT = 'C:/Python27/motionLearning/'
sys.path.append(ROOT)
import mylib

# モデルの読み込み
prop  = mylib.property.readProperty('property.txt')
model = mylib.NN.MyChain(*prop['nums'], bias=False)
serializers.load_npz('my.model', model)

DIV = 5 
# 中間層に直接入力した出力データを入力した際に出力させる値の違いを記録する
def saveDiffVector(fileName, model):
    fVector = open(fileName, 'w')
    fVector.write('# x\t' + 'y\t' + 'z\t' + 'dx\t' + 'dy\t' + 'dz\n')
    
    middleNum = len(model.layers) / 2
    dim = model.layers[middleNum]
    for z in mylib.util.drange(0.0, 1.0 if dim >= 3 else 0.0, 1.0 / DIV):
        for y in mylib.util.drange(0.0, 1.0 if dim >= 2 else 0.0, 1.0 / DIV):
            for x in mylib.util.drange(0.0, 1.0, 1.0 / DIV):
                start = [x, y, z]
                start = Variable(np.array([start[:dim]]).astype(np.float32))
                out = model.directActivate(middleNum, start)
                model(out)
                end = model.getMiddleValue()
                vec = np.concatenate((start.data[0], end.data[0] - start.data[0]), axis=0)
                fVector.write('\t'.join([str(v) for v in vec]) + '\n')
    fVector.close()

saveDiffVector('vector.dat', model)

