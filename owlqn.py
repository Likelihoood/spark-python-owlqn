#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
File: owlqn.py
Author: ricky.zhiyang(ricky.zhiyang@gmail.com)
Date: 2017/03/16 18:41:57
"""
import func
import sys
from func import performance
import pyspark

class OWLQN:
    def __init__(self,
            feaNum,
            regWeight = 0.01,
            tol = 0.001,
            iterNum = 100,
            intercept=True,
            m = 10):
        """
            brief info for: __init__

            Args:
                self :
                feaNum : 特征维数
                regWeight : 正则化系数
                tol : 迭代收敛的损失阈值
                iterNum : 最大迭代数
                intercept : 是否加入wx + b 中b
            Return:
            Raise:
        """
        self.regWeight = regWeight
        self.tol = tol
        self.iterNum = iterNum
        self.needIntercept = intercept
        if self.needIntercept:
            self.feaNum = feaNum + 1
        else:
            self.feaNum = feaNum
        self.m = m
        self.lossList = []

    @performance
    def train(self, sc, dataRdd):
        """
            brief info for: train
            训练LR模型， 返回模型的权重
            Args:
                self :
                dataRdd : [(label, dataDic)
                        lable: 1/-1 或者 1/0, 1: 正例, -1/0 : 负例
            Return: weights, 如果intercept为True,  {'intercept' : 值 }
            Raise:
        """
        if self.needIntercept:
            #这里为intercept默认增加一个维度的特征，即常数项b
            dataRdd = dataRdd.map(lambda item: func.addIntercept(item, self.feaNum))
            dataRdd.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
        it = 0
        #利用字典存储权重， key 为维度index, 仅记录非0值
        weight = {}
        #利用字典存储一阶梯度, key 为维度index, 仅存储非0值
        gradient = {}
        loss = 0.0
        sList = []
        roList = []
        yList = []
        alphaList = [0] * self.m
        #初始化计算loss 和 一阶梯度
        X = sc.broadcast(weight)
        loss, gradient = self.calLossAndGradient(dataRdd, X)
        #print "loss: %s" % loss
        #print "gradient: %s" % gradient
        self.firstLoss = loss
        self.lossList.append(loss)
        while it < self.iterNum:
            #print 'weight:', weight
            #print 'gradient:', gradient
            print "============iterator : %s ==========" % it
            # 1. 计算虚梯度
            vGradient = func.virtualGradient(self.feaNum, weight, gradient, self.regWeight)
            #print "vg: %s" % vGradient
            #print 'vg:', vGradient
            # 2. 保存虚梯度方向，用于后续确定搜索方向是否跨象限
            deepestDir = vGradient.copy()
            # 3. 利用LBFGS算法的两个循环计算下降方向, 这里会直接修改vGradient
            func.LBFGS(self.feaNum, vGradient, sList, roList, yList, alphaList)
            #print "LBFGS vg: %s" % vGradient
            #print 'vg:', vGradient
            # 4. 确定下降方向是否跨象限， 这里也会直接修改vGradient
            func.fixDirection(vGradient, deepestDir)
            #print "fixDirection vg: %s" % vGradient
            #print 'vg:', vGradient
            newLoss, newGradient, newWeight = self.backTrackingLineSearch(sc, it, loss, dataRdd, weight, gradient, vGradient)
            if self.check(it, loss, newLoss, newWeight):
                break
            else:
                self.shift(sList, yList, roList, weight, newWeight, gradient, newGradient)
                loss = newLoss
                gradient = newGradient
                weight = newWeight
            print "============iterator : %s end ==========" % it
            print ""
            it += 1
        self.newWeight = newWeight
        self.newLoss = newLoss
        self.finalItCount = it
        return newWeight

    def outputInfo(self, out, auc, pr):
        print >> out,  ""
        print >> out,  "============train finished =========="
        print >> out,  "paramCount : %s" %  self.feaNum
        print >> out,  "intercept: %s" % self.newWeight.get(self.feaNum-1, 0)
        it = self.finalItCount
        firstLoss = self.firstLoss
        newLoss = self.newLoss

        nonZeroCount = 0
        for index in self.newWeight:
            if self.newWeight[index] != 0:
                nonZeroCount += 1
        print >> out,  "non Zeron param Count: %s" % nonZeroCount
        print >> out,  "self.L1gweight: %s" % self.regWeight
        print >> out,  "tol: %s" % self.tol
        print >> out,  "total it count: %s, start loss is : %s, end loss is: %s, reduce loss is %s:" % (it, firstLoss, newLoss, firstLoss - newLoss)

        print >> out,  "AUC is: %s" % auc
        print >> out,  "PR is : %s" % pr
        print >> out,  "============model info finished========="
        print >> out,  ""

    @performance
    def calLossAndGradient(self, dataRdd, X):
        """
            brief info for: calLossAndGradient
            计算LR的loss function值
            Loss(w) = Sum_i(log(1 + exp(-y_i W * X_i))) + lambda * |W|
            Args:
                self :
                dataRdd : 数据
            Return:
            Raise:
        """
        beginValue = (0, {})
        rdd = dataRdd.map(lambda ins: func.lossAndGradient(ins, X))
        loss, gradient = rdd.fold(beginValue, func.sumLossAndGradient)
        w = 0
        for index in X.value:
            w += abs(X.value[index])
        loss = loss + self.regWeight * w
        return loss, gradient

    @performance
    def backTrackingLineSearch(self, sc, it, oldLoss, dataRdd, weight, gradient, vGradient):
        origDirDeriv = func.dirDeriv(self.regWeight, self.feaNum, weight, gradient, vGradient)
        #print "origDirDeriv : %s" % origDirDeriv
        if origDirDeriv >= 0:
            print >> sys.stderr, "oD: %s, L-BFGS chose a non-descent direction: check your gradient!" % origDirDeriv
            sys.exit(1)
        alpha = 1.0
        backoff = 0.5
        if it == 0:
            normalDir = func.dotValue(vGradient, vGradient) ** 0.5
            alpha =  1.0 / normalDir
            backoff = 0.1
        c1 = 1e-4
        while True:
            #print >> sys.stdout, "start to backline search alpha: %s" % alpha
            newWeight = func.getNewWeight(self.feaNum, weight, vGradient, alpha, self.regWeight)
            newX = sc.broadcast(newWeight)
            #print "newWeight: %s" % newWeight
            loss, newGradient = self.calLossAndGradient(dataRdd, newX)
            #print "oldLoss:%s , c1: %s, orgDirDeriv: %s, alpha: %s, loss:%s" % (oldLoss, c1, origDirDeriv, alpha, loss)
            newX.unpersist()
            if loss <= (oldLoss + c1 * origDirDeriv * alpha):

                return loss, newGradient, newWeight
            alpha *= backoff

    @performance
    def shift(self, sList, yList, roList, weight, newWeight, gradient, newGradient):
        size = len(sList)
        if size == self.m:
            #print >> sys.stdout, "pop 老的S, Y, RO"
            sList.pop(0)
            yList.pop(0)
            roList.pop(0)

        nextS = {}
        nextY = {}
        func.addMultInto(self.feaNum, nextS, newWeight, weight, -1)
        #print "newG: %s" % newGradient
        func.addMultInto(self.feaNum, nextY, newGradient, gradient, -1)
        #print "nextS: %s" % nextS
        #print "nextY: %s" % nextY
        ro = func.dotValue(nextS, nextY)
        sList.append(nextS)
        yList.append(nextY)
        roList.append(ro)

    def check(self, it, loss, newLoss, newWeight):
        if len(self.lossList) <= 5:
            print "please wait for more than 5 times iterator"
            self.lossList.append(newLoss)
            return False
        firstLoss = self.lossList[0]

        lastLoss = newLoss
        reduceLoss = (firstLoss - lastLoss )
        averageReduce = reduceLoss / len(self.lossList)

        reduceRatio = averageReduce / newLoss
        intercept = newWeight.get(self.feaNum - 1, "NOT_FOUND")
        if len(self.lossList) == 10:
            self.lossList.pop(0)
        self.lossList.append(lastLoss)

        print >> sys.stdout, "iterator: %s, oldLoss: %s, newLoss: %s, reduceLoss: %s , reduceRatio:%s, intercept: %s" % (it, firstLoss, lastLoss, averageReduce, reduceRatio, intercept)

        if reduceRatio <= self.tol:
            return True
        else:
            return False
