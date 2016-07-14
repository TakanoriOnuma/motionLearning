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

