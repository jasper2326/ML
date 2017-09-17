# author = Jasper_Jiao@ele.me
# -*- coding: cp936 -*-
# coding: cp936

import bayes

listPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listPosts)
# print myVocabList

# print bayes.setOfWord2Vec(myVocabList, listPosts[0])

trainMat = []
for postinDoc in listPosts:
    trainMat.append(bayes.setOfWord2Vec(myVocabList, postinDoc))

p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
# print p0V, p1V, pAb

bayes.testingNB()