# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 23:49
# @Author  : quincyqiang
# @File    : predict.py
# @Software: PyCharm
from keras.preprocessing.sequence import pad_sequences
from utils import bulid_dataset
test_sentence = ["Hawking", "was", "a", "Fellow", "of", "the", "Royal", "Society", ",", "a", "lifetime", "member",
                 "of", "the", "Pontifical", "Academy", "of", "Sciences", ",", "and", "a", "recipient", "of",
                 "the", "Presidential", "Medal", "of", "Freedom", ",", "the", "highest", "civilian", "award",
                 "in", "the", "United", "States", "."]

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=max_len)