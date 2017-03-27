#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: func.py
Author: ricky.zhiyang(ricky.zhiyang@gmail.com)
Date: 2017/03/16 19:14:50
"""
import math
import time

def performance(f): #定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    def fn(*args, **kw):       #对传进来的函数进行包装的函数
        t_start = time.time()  #记录函数开始时间
        r = f(*args, **kw)     #调用函数
        t_end = time.time()    #记录函数结束时间
        print 'call %s() in %fs' % (f.__name__, (t_end - t_start))  #打印调用函数的属性信息，并打印调用函数所用的时间
        return r               #返回包装后的函数
    return fn #调用包装后的函数

def dotValue(dataDic, weights):
    values = 0.0
    for index in dataDic:
        values += dataDic[index] * weights.get(index, 0)
    return values

def addMult(paramCount, vecDic1, vecDic2, c):
    for index in range(0, paramCount):
        v1 = vecDic1.get(index, 0)
        vecDic1[index] = v1 + vecDic2.get(index, 0) * c

def addMultInto(paramCount, vec1, vec2, vec3, c):
    for index in range(0, paramCount):
        vec1[index] = vec2.get(index, 0) + vec3.get(index, 0) * c

def scale(vecDic1, c):
    for index in vecDic1:
        vecDic1[index] *= c

def addIntercept(item, feaNum):
    """
        brief info for: addIntercept
        向每个样本中增加一个常数项，用来计算b的大小
        Args:
            item :
            feaNum :
        Return:
        Raise:
    """
    label, dataDic = item
    dataDic[feaNum - 1] = 1.0
    return (label, dataDic)

def scoreOfIns(ins, weight):
    """
        brief info for: scoreOfIns
        单个样本的y_i * (WX + b)
        Args:
            ins : {feature_index: x_value}
            weight : {feature_index: w_value}
        Return:  y * (wx + b)
        Raise:
    """
    score = 0.0
    label, dataDic = ins
    for index in dataDic:
        x = dataDic[index]
        w = weight.get(index, 0)
        score += w * x
    if label <= 0:
        score = -1 * score
    return score

def lossAndGradient(ins, X):
    """
        brief info for: lossAndGradient
        计算单样本的loss 和 gradient
        loss = log(1 + exp(-y * wx))
        gradient_k = Sum(1/(1 + exp(-ywx)) * y * x_k)
        这里仅仅计算Sum中的部分， 但是是对该ins来说， 它对每个维度的gradient都算一次
        Args:
            ins :
            weights :
        Return: loss, gradient
                loss为一个ins的loss
                gradient是该ins对每个维度的贡献值
        Raise:
    """
    weights = X.value
    score = scoreOfIns(ins, weights)
    insLoss = 0
    insProb = 0
    if score < -30:
        insLoss = -score
        insProb = 0
    elif score > 30:
        insLoss = 30
        insProb = 1
    else:
        temp = 1.0 + math.exp(-score)
        insLoss  = math.log(temp, math.e)
        insProb = 1.0 / temp
    gradient = {}
    label, dataDic = ins
    insProb = insProb - 1
    if label <= 0:
        insProb = -1 * insProb
    for index in dataDic:
        gradient[index] = insProb * dataDic[index]

    return insLoss, gradient

def sumLossAndGradient(dataPair1, dataPair2):
    loss1, gradient1 = dataPair1
    loss2, gradient2 = dataPair2
    loss = loss1 + loss2

    for index in gradient1:
        gradient2.setdefault(index, 0)
        gradient2[index] += gradient1[index]
    return loss, gradient2

def lossValue(ins, weights):
    """
        brief info for: lossValue
        计算单个样本的lossFunction 的值
        log(1 + exp(-y * w*x)) + lambda * |W|
        Args:
            ins :(label, dataDic)
            weight : 权重向量数据
        Return: double loss value
        Raise:
    """
    label = ins[0]
    dataDic = ins[1]
    loss = dotValue(dataDic, weights)
    if not label:
        loss = -1 * loss
    if loss < -30:
        return -loss
    elif loss > 30:
        return 0
    else:
        loss = math.log(1 + math.exp(-loss), math.e)
        return loss

def findVG(w, g, l1weight):
    """
        brief info for: findVG
        依据 w 和 对应的梯度， 确定虚梯度的负方向
        Args:
            w : 当前参数值
            g : 当前参数值的 梯度值
            l1weight : l1 正则化系数
        Return:   虚梯度的负方向
        Raise:
    """
    if w < 0:
        return -g + l1weight
    elif w > 0:
        return -g - l1weight
    else : #g==0
        if g < (-l1weight):
            return -g - l1weight
        elif g > l1weight:
            return -g + l1weight
        else:
            return 0

@performance
def virtualGradient(paramCount, weight, gradient, l1weight):
    """
        brief info for: virtualGradient
        依据梯度确定虚梯度的负方向
        Args:
            weight:
            gradient :
            l1weight :
        Return:
        Raise:
    """
    vGradient = {}
    if l1weight == 0:
        for index in gradient:
            vGradient[index] = gradient[index] * -1
        return vGradient

    for index in range(0, paramCount):
        g = gradient.get(index, 0)
        w = weight.get(index, 0)
        vGradient[index] = findVG(w, g, l1weight)
    return vGradient

@performance
def LBFGS(paramCount, vGradient, sList, roList, yList, alphaList):
    """
        brief info for: LBFGS
        LBFGS的两个循环, 求解Hessain的逆 H 和 虚梯度负方向的乘积
        也就是新的参数的下降方向， - H * f'
        Args:
            paramCount : 参数数量
            vGradient : 虚梯度的负方向
            sList : S
            roList : rou
            yList : Y
            alphaList : alpha
        Return:
        Raise:
    """
    count = len(sList)
    if count > 0:
        indexList = range(0, count)
        indexList.reverse()
        for i in indexList:
            alphaList[i] = -1.0 * dotValue(sList[i], vGradient) / roList[i]
            addMult(paramCount, vGradient, yList[i], alphaList[i])

        lastY = yList[-1]
        yDotY = dotValue(lastY, lastY)
        scalar = roList[-1] / yDotY
        scale(vGradient, scalar);

        for i in range(0, count):
            beta = dotValue(yList[i], vGradient) / roList[i]
            addMult(paramCount, vGradient, sList[i], -alphaList[i] - beta);

@performance
def fixDirection(vg, rawVG):
    """
        brief info for: fixDirection
        计算下降方向在虚梯度方向的投影
        如果方向和原始虚梯度负方向不一致，则直接置零
        Args:
            vg : 参数下降方向
            rawVG : 原始虚梯度方向
        Return:
        Raise:
    """
    for index in vg:
        if vg[index] * rawVG[index] <= 0:
            vg[index] = 0

def dirDeriv(l1weight, paramCount, weight, gradient, vGradient):
    """
        brief info for: dirDeriv
        确定虚梯度方向累加修正值，主要用户BackLineSearch
        Args:
            l1weight :
            weight : 参数dict
            vgradient : 虚梯度
        Return:
        Raise:
    """

    if l1weight == 0:
        return dotValue(gradient, vGradient)
    else:
        value = 0.0
        for i in range(0, paramCount):
            vg = vGradient.get(i, 0)
            w = weight.get(i, 0)
            g = gradient.get(i, 0)
            if vg != 0:
                if w < 0:
                    value += vg * (g - l1weight)
                elif w > 0:
                    value += vg * (g + l1weight)
                elif vg < 0:
                    value += vg * (g - l1weight)
                elif vg > 0:
                    value += vg * (g + l1weight)
        return value

def getNewWeight(paramCount, weight, vGradient, alpha, l1weight):
    """
        brief info for: getNewWeight
        计算新的权重
        Args:
            paramCount : 参数数量
            weight : 参数当前权重dict
            vGradient : 虚梯度方向
            alpha : 步长
            l1weight : l1 系数
        Return: 新的参数
        Raise:
    """
    newWeight = {}
    addMultInto(paramCount, newWeight, weight, vGradient, alpha)
    if l1weight <= 0:
        return newWeight

    for index in range(0, paramCount):
        if newWeight.get(index, 0) * weight.get(index, 0) < 0:
            newWeight[index] = 0.0

    return newWeight

def filterZeroValue(dic):
    newDic = {}
    for index in dic:
        if dic[index] != 0.0:
            newDic[index] = dic[index]
    #print "old_dic: %s VS new_dic: %s" % (len(dic), len(newDic))
    return newDic
