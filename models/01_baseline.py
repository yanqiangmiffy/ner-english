# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 23:49
# @Author  : quincyqiang
# @File    : 01_memorization_baseline.py
# @Software: PyCharm
from utils import load_data

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

getter=SentenceGetter(data)
sent,pos,tag=getter.get_next()
print(sent); print(pos); print(tag)

#