# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 23:34
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def load_data(filename):
    data=pd.read_csv(filename,encoding="latin1")
    data=data.fillna(method="ffill") # 用前一个非缺失值去填充该缺失值
    # print(data.head(10))
    # words=list(data['Word'].values)
    words=list(set(data['Word'].values))
    n_words=len(words)
    # print(n_words)
    return data


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

# if __name__ == '__main__':
#     data_dir='data/ner_dataset.csv'
#     data=load_data(data_dir)