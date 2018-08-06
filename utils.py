# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 23:34
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def load_data(filename):
    data=pd.read_csv(filename,encoding="latin1")
    data=data.fillna(method="ffill")
    # print(data.tail(10))
    # words=list(data['Word'].values)
    words=list(set(data['Word'].values))
    n_words=len(words)
    print(n_words)
    return data


# if __name__ == '__main__':
#     data_dir='data/ner_dataset.csv'
#     data=load_data(data_dir)