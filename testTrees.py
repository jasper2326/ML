# author = Jasper_Jiao@ele.me
# -*- coding: cp936 -*-
# coding: cp936

import trees

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]

    label = ['no surfacing', 'flippers']
    return dataSet, label

mydata, labels = createDataSet()
print mydata
print trees.calcShannonEnt(mydata)

# mydata[0][-1] = 'maybe'
# print mydata
# print trees.calcShannonEnt(mydata)

print trees.splitDataSet(mydata, 1, 1)
print trees.splitDataSet(mydata, 1, 0)

print trees.chooseBestFeatureToSplit(mydata)

print trees.createTree(mydata, labels)