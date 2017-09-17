# author = Jasper_Jiao@ele.me
# -*- coding: cp936 -*-
# coding: cp936

import numpy
import math

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1 + math.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    m, n = numpy.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = numpy.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    m, n = numpy.shape(dataMatrix)
    alpha = 0.1
    weights = numpy.ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i] * weights)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]


def stocGradAscent0(dataMatrix, classLabels, numIter = 150):
    m, n = numpy.shape(dataMatrix)
    weights = numpy.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + i + j) + 0.01
            randIndex = int(numpy.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights


