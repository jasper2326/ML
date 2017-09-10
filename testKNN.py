# author = Jasper_Jiao@ele.me
# -*- coding: cp936 -*-
# coding: cp936

import KNN

group, lables = KNN.createDataSet()
print(group, lables)

print(KNN.classify0([10,0], group, lables, 3))