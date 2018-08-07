import numpy as np
import pandas as pd
from utils import load_data,SentenceGetter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model,Input
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
plt.style.use("ggplot")

# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
data=load_data(ner_dataset_dir)
# 标签和单词
words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words)
tags = list(set(data["Tag"].values))
n_tags=len(tags)
getter=SentenceGetter(data)
sentences=getter.sentences
# print(sentences[0])

plt.hist([len(s) for s in sentences],bins=50)
plt.show()

# 2 构建数据集
# 输入长度等长，统一设置为50
max_len=50
word2idx={w:i for i,w in enumerate(words)}
tag2idx={t:i for i,t in enumerate(tags)}

# print(word2idx['Obama'])
# print(tag2idx['B-geo'])

# 填充句子
X=[[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
# print(X[1])

# 填充标签
y=[[tag2idx[w[2]] for w in s] for s in sentences]
y=pad_sequences(maxlen=max_len,sequences=y,padding="post",value=tag2idx["O"])
# print(y[1])

# 将label转为categorial
y=[to_categorical(i,num_classes=n_tags) for i in y]

# 划分数据集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
print(X_train.shape,np.array(y_test).shape)

# 3 构建和训练模型
input=Input(shape=(max_len,))
model=Embedding(input_dim=n_words,output_dim=50,input_length=max_len)(input)
model=Dropout(0.1)(model)
model=Bidirectional(LSTM(units=100,return_sequences=True,recurrent_dropout=0.1))(model)
out=TimeDistributed(Dense(n_tags,activation='softmax'))(model) # softmax output layer

model=Model(input,out)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history=model.fit(X_train,np.array(y_train),batch_size=32,epochs=5,validation_split=0.1,verbose=1)


hist = pd.DataFrame(history.history)
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()

# 4 预测
i = 2318
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test[i], -1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_test[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))