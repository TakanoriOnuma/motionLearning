#!/usr/bin/env python
# coding:utf-8

# 設定情報を記録する
def writeProperty(fileName, prop):
    f = open(fileName, 'w')    
    for key, value in prop.items():
        # 記録する必要のないプロパティはスキップする
        if key == 'nums' or key == 'midLayerNum':
            continue
        f.write('{}:{}\n'.format(key, value))
    f.close()

# 学習情報の読み込み
def readProperty(fileName):
    prop = {}
    fProp = open(fileName, 'r')
    for line in fProp.readlines():
        line = line[:-1]    # 改行を取る
        key, value = line.split(':')
        # 特別な処理
        if key == 'NN':
            prop['NN'] = value
            nums = [int(num) for num in value.split('-')]
            prop['nums'] = nums
            prop['middleNum'] = nums[len(nums) / 2]

        # 一般的な処理
        # int型ならintに変換して代入
        if value.isdigit():
            prop[key] = int(value)
        else:
            # floatの変換を行い、失敗したらstr型のまま代入
            try:
                prop[key] = float(value)
            except:
                prop[key] = value       
    fProp.close()
    return prop
