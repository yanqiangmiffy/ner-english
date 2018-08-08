import numpy as np
from keras.preprocessing.sequence import pad_sequences
from models.utils import load_data,SentenceGetter,bulid_dataset
from keras.models import Model,Input,load_model
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from flask import Flask,request,render_template,jsonify
import keras
keras.backend.clear_session()
app=Flask(__name__)


# 1 加载数据
ner_dataset_dir='data/ner_dataset.csv'
dataset_dir='data/dataset.pkl'
data=load_data(ner_dataset_dir)

# getter=SentenceGetter(data)
# sentences=getter.sentences
# sentences=[" ".join([w[0] for w in s ]) for s in sentences]

# 2 构建数据
n_words, n_tags, max_len, words,tags,\
X_train, X_test, y_train, y_test=bulid_dataset(ner_dataset_dir,dataset_dir,max_len=50)
word2idx = {w: i for i, w in enumerate(words)}


# 3 加载模型
bilstm_model = load_model(filepath="result/bi-lstm.h5")

# load 进来模型紧接着就执行一次 predict 函数
print('test train...')
bilstm_model.predict(np.zeros((1, 50)))


input = Input(shape=(max_len,))
bilstm_crf_model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
bilstm_crf_model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(bilstm_crf_model)  # variational biLSTM
bilstm_crf_model = TimeDistributed(Dense(50, activation="relu"))(bilstm_crf_model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(bilstm_crf_model)  # output
bilstm_crf_model = Model(input, out)
save_load_utils.load_all_weights(bilstm_crf_model, filepath="result/bilstm-crf.h5")


bilstm_crf_model.predict(np.zeros((1, 50)))
print('test done.')

# 测试数据
def build_input(test_sentence):
    test_sentence =test_sentence.split(" ")
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                                    padding="post", value=0, maxlen=max_len)
    return test_sentence,x_test_sent


def bilstm_predcit(model,test_sentence,x_test_sent):
    pred = model.predict(np.array([x_test_sent[0]]))
    pred = np.argmax(pred, axis=-1)

    temp = []
    for _,p in zip(test_sentence,pred[0]):
        temp.append(tags[p])

    # result = {
    #     "method":"Bi-directional LSTM",
    #     "sentence": " ".join(test_sentence),
    #     "tags": " ".join(temp)
    # }
    result = {
        "method": "Bi-directional LSTM",
        "sentence": test_sentence,
        "tags": temp
    }
    return result


def bilstm_crf_predcit(model,test_sentence,x_test_sent):
    pred = model.predict(np.array([x_test_sent[0]]))
    pred = np.argmax(pred, axis=-1)

    temp = []
    for _, p in zip(test_sentence, pred[0]):
        temp.append(tags[p])

    # result = {
    #     "method": "Bi-directional LSTM+CRF",
    #     "sentence": " ".join(test_sentence),
    #     "tags":" ".join(temp)
    # }
    result = {
        "method": "Bi-directional LSTM+CRF",
        "sentence": test_sentence,
        "tags": temp
    }
    return result

@app.route('/ner',methods=['post','get'])
def ner():
    origin_test_sentence=request.args.get('test_sent',type=str)
    print(origin_test_sentence)
    test_sentence, x_test_sent=build_input(origin_test_sentence)

    result1=bilstm_predcit(bilstm_model,test_sentence, x_test_sent)
    result2=bilstm_crf_predcit(bilstm_crf_model,test_sentence,x_test_sent)

    return jsonify(result=[result1,result2])

@app.route('/named-entity-recognition')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)