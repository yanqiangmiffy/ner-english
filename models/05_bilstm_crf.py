import argparse
import numpy as np
import pandas as pd
from utils import bulid_dataset
from keras.models import Model,Input
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

# import matplotlib.pyplot as plt
# plt.style.use("ggplot")

# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
dataset_dir='../data/dataset.pkl'

# 2 构建数据集
n_words, n_tags, max_len, words,tags,\
X_train, X_test, y_train, y_test=bulid_dataset(ner_dataset_dir,dataset_dir,max_len=50)

def train():
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)


    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=1,
                        validation_split=0.1, verbose=1)
    save_load_utils.save_all_weights(model,filepath="../result/bilstm-crf.h5")

    # hist = pd.DataFrame(history.history)
    # plt.figure(figsize=(12,12))
    # plt.plot(hist["acc"])
    # plt.plot(hist["val_acc"])
    # plt.show()

def sample():
    """
    利用已经训练好的数据进行预测
    :return:
    """
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
    save_load_utils.load_all_weights(model,filepath="../result/bilstm-crf.h5")

    # 预测
    i = 300
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_test[i], -1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X_test[i], true, p[0]):
        print("{:15}: {:5} {}".format(words[w], tags[t], tags[pred]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="命名执行训练或者预测")
    parser.add_argument('--action', required=True, help="input train or test")
    args = parser.parse_args()
    if args.action == 'train':
        train()
    if args.action == 'test':
        sample()
