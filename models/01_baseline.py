# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 23:49
# @Author  : quincyqiang
# @File    : 01_memorization_baseline.py
# @Software: PyCharm
from models.utils import load_data

# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
data=load_data(ner_dataset_dir)


# 2 构建数据
class SentenceGetter(object):


    def __init__(self,data):
        self.n_sent=1
        self.data=data
        self.empty=False

    def get_next(self):
        try:
            s=self.data[self.data['Sentence #']=="Sentence: {}".format(self.n_sent)]
            self.n_sent+=1
            return s['Word'].tolist(),s['POS'].tolist(),s['Tag'].tolist()
        except:
            self.empty=True
            return None,None,None

# getter=SentenceGetter(data)
# sent,pos,tag=getter.get_next()
# print(sent); print(pos); print(tag)

# 3 方法1：记住最常见的实体，不知道的实体我们就预测为“O”
from sklearn.base import BaseEstimator,TransformerMixin
class MemoryTagger(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        """
        words为X，tags为y
        :param X:
        :param y:
        :return:
        """
        voc={}
        self.tags=[]
        for x,t in zip(X,y):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t]+=1
                else:
                    voc[x][t]=1
            else:
                voc[x]={t:1}
        self.memory={}
        for k,d in voc.items():
            self.memory[k]=max(d,key=d.get) #  获取出现次数最多的标签
    def predict(self,X,y=None):
        """
        从memory预测tag，如果word是未知的，则预测为'O'
        :param X:
        :param y:
        :return:
        """
        return [self.memory.get(x,'O') for x in X]

# tagger=MemoryTagger()
# tagger.fit(sent,tag)
# print(tagger.predict(sent))
# print(tagger.tags)
# print(tagger.memory)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
words=data['Word'].tolist()
tags=data["Tag"].tolist()
pred=cross_val_predict(estimator=MemoryTagger(),X=words,y=tags,cv=5)
report=classification_report(y_pred=pred,y_true=tags)
print(report)