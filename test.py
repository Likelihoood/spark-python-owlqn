#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: train.py
Author: ricky.zhiyang(ricky.zhiyang@gmail.com)
Date: 2017/03/20 17:08:46
"""
import pyspark
from owlqn import OWLQN
import math

def genEnv():
    conf = pyspark.SparkConf().setAppName("spark-owlqn")
    sc = pyspark.SparkContext(conf = conf)
    return sc

def genTestData(sc):
    data = [
        (1, {1: 1.0, 2:1.0, 3:1.0, 5:1.0, 8:1.0}),
        (1, {1: 1.0, 3: 1.0, 4:1.0, 7: 1.0}),
        (-1, {2:1.0, 5:1.0, 9:1.0}),
        (-1, {0:1.0, 3:1.0, 4:1.0, 5:1.0}),
        (-1, {0:1.0, 2:1.0, 3:1.0, 6:1.0, 7:1.0}),
        (-1, {3:1.0, 6:1.0}),
        (-1, {0:1.0, 2:1.0, 7:1.0}),
        (-1, {6:1.0, 7:1.0}),
        (1, {1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 8:1.0}),
        (1, {1:1.0, 3:1.0, 5:1.0, 6:1.0}),
        (-1, {0:1.0, 2:1.0, 9:1.0}),
        (-1, {0:1.0, 4:1.0, 7:1.0})
    ]
    rdd = sc.parallelize(data)
    out = open('train_data', 'w')
    for d in data:
        label = d[0]
        dic = d[1]
        outList = []
        for index in dic:
            outList.append('%s:%s' % (index, dic[index]) )
        outStr = ','.join(outList)
        outStr = '%s\t%s\n' % (label, outStr)
        out.write(outStr)
    out.close()


    return rdd, data, 10

def calLr(insDic, model, feaNum):
    w = model.get(feaNum, 0)
    for index in insDic:
        w += insDic[index] * model.get(index)
    y = 1 / (1 + math.exp(-w))
    return y

def predict(data, model, feaNum):
    label = data[0]
    insDic = data[1]
    predictY = calLr(insDic, model, feaNum)
    return label, predictY

def testTrain(sc):
    rdd, dataList, feaNum = genTestData(sc)
    lrm = OWLQN(feaNum)
    model = lrm.train(sc, rdd)

    for index in model:
        print index, model[index]

    for data in dataList:
        label, predictY = predict(data, model, feaNum)
        print label, predictY

if __name__ == '__main__':
    sc = genEnv()
    testTrain(sc)
