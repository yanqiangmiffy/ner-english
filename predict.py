# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 23:49
# @Author  : quincyqiang
# @File    : predict.py
# @Software: PyCharm
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from models.utils import bulid_dataset
from keras.models import Model,Input,load_model
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils


# 1 加载数据
ner_dataset_dir='data/ner_dataset.csv'
dataset_dir='data/dataset.pkl'

# 2 加载数据
n_words, n_tags, max_len, words,tags,\
X_train, X_test, y_train, y_test=bulid_dataset(ner_dataset_dir,dataset_dir,max_len=50)
word2idx = {w: i for i, w in enumerate(words)}

# 测试数据
test_sentence = ["Hawking", "was", "a", "Fellow", "of", "the", "Royal", "Society", ",", "a", "lifetime", "member",
                     "of", "the", "Pontifical", "Academy", "of", "Sciences", ",", "and", "a", "recipient", "of",
                     "the", "Presidential", "Medal", "of", "Freedom", ",", "the", "highest", "civilian", "award",
                     "in", "the", "United", "States", "."]

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                                padding="post", value=0, maxlen=max_len)


def bilstm_predcit():
    model = load_model(filepath="result/bi-lstm.h5")
    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)

    print("{:15}||{}".format("Word", "Prediction"))
    print(30 * "=")
    for w, pred in zip(test_sentence, p[0]):
        print("{:15}: {:5}".format(w, tags[pred]))


def bilstm_crf_predcit():

    # 重新初始化模型，构建配置信息，和train部分一样
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)

    # 恢复权重
    save_load_utils.load_all_weights(model, filepath="result/bilstm-crf.h5")

    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    print("{:15}||{}".format("Word", "Prediction"))
    print(30 * "=")
    for w, pred in zip(test_sentence, p[0]):
        print("{:15}: {:5}".format(w, tags[pred]))

if __name__ == '__main__':
    bilstm_predcit()
    bilstm_crf_predcit()