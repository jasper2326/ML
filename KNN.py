# author = Jasper_Jiao@ele.me
# -*- coding: cp936 -*-
# coding: cp936

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def classify0(inX, dataSet, lables, k):
    # shape: read array's length
    dataSetSize = dataSet.shape[0]
    # A沿着各个维度重复次数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 求平方
    sqDiffMat = diffMat ** 2
    # 求平方和
    sqDistance = sqDiffMat.sum(axis=1)
    # 得到距离
    distance = sqDistance ** 0.5
    # 得到排序后的元素下标
    sortedDistIndices = distance.argsort()
    # 新建储存排序结果的字典
    classCount = {}
    for i in range(k):
        # 遍历lable
        voteIlable = lables[sortedDistIndices[i]]
        # 计算距离最小的k个点
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回最大值
    return sortedClassCount[0][0]


def file2Matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLableVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index: ] = listFromLine[0:3]
        classLableVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLableVector

