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
    # A���Ÿ���ά���ظ�����
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # ��ƽ��
    sqDiffMat = diffMat ** 2
    # ��ƽ����
    sqDistance = sqDiffMat.sum(axis=1)
    # �õ�����
    distance = sqDistance ** 0.5
    # �õ�������Ԫ���±�
    sortedDistIndices = distance.argsort()
    # �½��������������ֵ�
    classCount = {}
    for i in range(k):
        # ����lable
        voteIlable = lables[sortedDistIndices[i]]
        # ���������С��k����
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    # ����
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # �������ֵ
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

