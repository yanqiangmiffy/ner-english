import argparse
import numpy as np
import pandas as pd
from utils import bulid_dataset
import matplotlib.pyplot as plt
from keras.models import Model,Input,load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
plt.style.use("ggplot")

# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
dataset_dir='../data/dataset.pkl'

# 2 构建数据集
n_words, n_tags, max_len, words,tags,\
X_train, X_test, y_train, y_test=bulid_dataset(ner_dataset_dir,dataset_dir,max_len=50)

# 3 构建和训练模型
def train():
    input=Input(shape=(max_len,))
    model=Embedding(input_dim=n_words,output_dim=50,input_length=max_len)(input)
    model=Dropout(0.1)(model)
    model=Bidirectional(LSTM(units=100,return_sequences=True,recurrent_dropout=0.1))(model)
    out=TimeDistributed(Dense(n_tags,activation='softmax'))(model) # softmax output layer

    model=Model(input,out)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    # checkpoint
    # filepath = "../result/bilstm-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # history=model.fit(X_train,np.array(y_train),batch_size=32,epochs=5,validation_split=0.1,verbose=1,callbacks=[checkpoint])

    history=model.fit(X_train,np.array(y_train),batch_size=32,epochs=5,validation_split=0.1,verbose=1)
    # 保存模型
    model.save(filepath="../result/bi-lstm.h5")


    hist = pd.DataFrame(history.history)
    plt.figure(figsize=(12,12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()

def sample():
    """
    利用已经训练好的数据进行预测
    :return:
    """
    model=load_model(filepath="../result/bi-lstm.h5")
    # 4 预测
    i = 300
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_test[i], -1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X_test[i], true, p[0]):
        print("{:15}: {:5} {}".format(words[w], tags[t], tags[pred]))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="命名执行训练或者预测")
    parser.add_argument('--action',required=True,help="input train or test")
    args=parser.parse_args()
    if args.action=='train':
        train()
    if args.action=='test':
        sample()
