#!/usr/bin/env python
# coding:utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList

class MyChain(ChainList):
    def __init__(self, *layers, **options):
        super(MyChain, self).__init__()
        opt = {
            'beta' : 0.5,
            'bias' : False,
            'geneFuncW' : np.random.randn
        }
        for key in options:
            if key not in opt.keys():
                print 'undefined key: {0}'.format(key)
            opt[key] = options[key]

        self.beta = opt['beta']
        self.layers = layers
        for i in range(len(layers) - 1):
            initW = opt['geneFuncW'](layers[i + 1], layers[i])
            self.add_link(L.Linear(layers[i], layers[i + 1], nobias=(not opt['bias']), initialW=initW))

    def __call__(self, x, train=False):
        self.value = [None] * (len(self) + 1)
        self.value[0] = x
        for i in range(len(self)):
            self.value[i + 1] = F.sigmoid(self.beta * self[i](self.value[i]))
        return self.value[-1]

    def directActivate(self, startLayer, x):
        self.value = [None] * (len(self) + 1)
        self.value[startLayer] = x
        for i in range(startLayer, len(self)):
            self.value[i + 1] = F.sigmoid(self.beta * self[i](self.value[i]))
        return self.value[-1]                                        

    def getMiddleValue(self):
        return self.value[len(self) / 2]

