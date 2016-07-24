# -*- coding: utf-8 -*-
import sys, os, cv2
import numpy as np
from chainer import Variable, serializers

sys.path.append('C:\Python27\motionLearning')
import mylib

DIV = 5 
# 中間層に直接入力した出力データを画像に保存する
def directActivateImages(rootDir, model):
    # フォルダを作成する
    if not os.path.exists(rootDir + '/images'):
        os.mkdir(rootDir + '/images')
    if not os.path.exists(rootDir + '/merge'):
        os.mkdir(rootDir + '/merge')
        
    middleNum = len(model.layers) / 2
    dim = model.layers[middleNum]
    for z in mylib.util.drange(0.0, 1.0 if dim >= 3 else 0.0, 1.0 / DIV):
        for y in mylib.util.drange(0.0, 1.0 if dim >= 2 else 0.0, 1.0 / DIV):
            for x in mylib.util.drange(0.0, 1.0, 1.0 / DIV):
                inp = [x, y, z]
                inp = Variable(np.array([inp[:dim]]).astype(np.float32))
                out = model.directActivate(middleNum, inp)
                img = mylib.image.makeOutputData(out.data, 100, 100)
                cv2.imwrite(rootDir + '/images/output({},{},{}).png'.format(x, y, z), img)

    # 中間層に直接入力した画像をマージする
    for z in mylib.util.drange(0.0, 1.0 if dim >= 3 else 0.0, 1.0 / DIV):
        imgRows = []
        for y in mylib.util.drange(0.0, 1.0 if dim >= 2 else 0.0, 1.0 / DIV):
            imgCols = []
            for x in mylib.util.drange(0.0, 1.0, 1.0 / DIV):
                fileName = rootDir + '/images/output({},{},{}).png'.format(x, y, z)
                imgCols.append(cv2.imread(fileName))
            imgRows.append(cv2.hconcat(imgCols))
        image = cv2.vconcat(imgRows[::-1])
        cv2.imwrite(rootDir + '/merge/output(z={})_{}.png'.format(z, DIV), image)

# モデルの読み込み
prop  = mylib.property.readProperty('property.txt')
model = mylib.NN.MyChain(*prop['nums'], bias=False)
serializers.load_npz('my.model', model)

# 記録する
if not os.path.exists('trained'):
    os.mkdir('trained')
directActivateImages('trained', model)
