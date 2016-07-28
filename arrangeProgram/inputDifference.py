# -*- coding: utf-8 -*-
import sys, os, cv2
import numpy as np
from chainer import Variable, serializers

ROOT = 'C:/Python27/motionLearning/'
sys.path.append(ROOT)
import mylib

# ノイズ入力による特徴層の出力の差を記録する
def writeDifferenceByNoiseInput(fileName, model, noiseNum, inpDataList, trainTypes):
    fDiff = open(fileName, 'w')
    fDiff.write('# trainType\t' + 'distProp_mean\t' + 'distProp_std' + '\n')
    for trainType in trainTypes:        
        noiseMean = []
        noiseStd  = []
        for inp in inpDataList[trainType]:
            # 基本のデータを入力し、特徴層の出力値を取得
            model(Variable(np.array([inp]).astype(np.float32)))
            mid = model.getMiddleValue().data[0]

            # ノイズを入れて、特徴層の出力値を取得
            noise = np.random.normal(0, 0.05, (noiseNum, len(inp)))
            inp_noise = np.clip(inp + noise, 0.1, 0.9)
            model(Variable(np.array(inp_noise).astype(np.float32)))
            mid_noise = model.getMiddleValue().data

            # 入力値でのノイズによる距離の差と特徴層でのノイズによる距離の差を求める
            inp_dist = np.linalg.norm(inp - inp_noise, axis=1)
            mid_dist = np.linalg.norm(mid - mid_noise, axis=1)

            # 特徴層の出力値と入力値の比を求める
            dist_prop = mid_dist / inp_dist
            noiseMean.append(dist_prop.mean())
            noiseStd.append(dist_prop.std())

        noiseMean = np.array(noiseMean)
        noiseStd  = np.array(noiseStd)
        fDiff.write('{}\t{}\t{}\n'.format(trainType, noiseMean.mean(), noiseStd.mean()))
    fDiff.close()

# モデルの読み込み
prop  = mylib.property.readProperty('property.txt')
model = mylib.NN.MyChain(*prop['nums'], bias=False)
serializers.load_npz('my.model', model)

if not prop.has_key('DATA_TYPE'):
    prop['DATA_TYPE'] = 'normal'

# 入力データの読み込み
trainTypes = ['train', 'normal-test', 'anomaly-test']
inpDataList = {}
for trainType in trainTypes:
    rootDir = ROOT + '{}/{}/{}'.format(prop['IMG_DIR'], prop['DATA_TYPE'], trainType)
    dataList, emphaList = mylib.image.createInputDataList(rootDir, prop['DATA_NUM'], prop['EMPHA_VALUE'])
    inpDataList[trainType] = np.array(dataList).astype(np.float32)

# ノイズ入力による特徴層の出力値の差を記録する
NOISE_NUM = 100
writeDifferenceByNoiseInput('noiseDiff.dat', model, NOISE_NUM, inpDataList, trainTypes)




