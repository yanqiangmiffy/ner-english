import numpy as np
import pandas as pd
from models.utils import load_data,bulid_dataset
from keras.models import Model,Input,load_model
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
from keras_contrib.layers import CRF


# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
data=load_data(ner_dataset_dir)


# 2 构建数据集
n_words, n_tags, max_len, words,tags,\
X_train, X_test, y_train, y_test=bulid_dataset(data)


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
history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5,
                    validation_split=0.1, verbose=1)
model.save(filepath="../result/bilstm-crf.h5")

i = 1927
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test[i], -1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_test[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))

