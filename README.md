# ner-english
:leopard: 英文命名实体识别(NER)的研究

## 准备
- 数据集：Kaggle-https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/4
- 词汇量：去重之后：`35178`
- 句子：`47959` 
- 实体标签含义：
```
geo = Geographical Entity 地名
org = Organization 组织
per = Person 人物
gpe = Geopolitical Entity 地理政治
tim = Time indicator 时间
art = Artifact 艺术
eve = Event 时间
nat = Natural Phenomenon 自然现象
```
## 模型
- 01_basline

  简单的标签统计特征
  ```text
               precision    recall  f1-score   support

      B-art       0.20      0.05      0.09       402
      B-eve       0.54      0.25      0.34       308
      B-geo       0.78      0.85      0.81     37644
      B-gpe       0.94      0.93      0.94     15870
      B-nat       0.42      0.28      0.33       201
      B-org       0.67      0.49      0.56     20143
      B-per       0.78      0.65      0.71     16990
      B-tim       0.87      0.77      0.82     20333
      I-art       0.04      0.01      0.01       297
      I-eve       0.39      0.12      0.18       253
      I-geo       0.73      0.58      0.65      7414
      I-gpe       0.62      0.45      0.52       198
      I-nat       0.00      0.00      0.00        51
      I-org       0.69      0.53      0.60     16784
      I-per       0.73      0.65      0.69     17251
      I-tim       0.58      0.13      0.21      6528
          O       0.97      0.99      0.98    887908

    avg / total       0.94      0.95      0.94   1048575
  ```
- 02_random_forest_classifier:

  基本特征：`首字母是否大写，是否小写，是否为大写，单词长度，是否为数字，是否全为字母`
  
  上下文特征:`上下文单词的标签以及词性特征`
  
  方法：`RandomForestClassifier`
  ```text
               precision    recall  f1-score   support

      B-art       0.19      0.08      0.11       402
      B-eve       0.39      0.25      0.30       308
      B-geo       0.81      0.85      0.83     37644
      B-gpe       0.98      0.93      0.95     15870
      B-nat       0.28      0.28      0.28       201
      B-org       0.71      0.60      0.65     20143
      B-per       0.84      0.73      0.78     16990
      B-tim       0.90      0.79      0.84     20333
      I-art       0.05      0.02      0.02       297
      I-eve       0.21      0.10      0.13       253
      I-geo       0.74      0.64      0.69      7414
      I-gpe       0.80      0.45      0.58       198
      I-nat       0.40      0.20      0.26        51
      I-org       0.69      0.65      0.67     16784
      I-per       0.81      0.74      0.78     17251
      I-tim       0.76      0.47      0.58      6528
          O       0.98      0.99      0.99    887908

    avg / total       0.95      0.96      0.95   1048575
  ```
- 03_CRF 条件随机场
  
  > 特征基本同上
  ```python
    crf=CRF(algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=False)    
  ```
  ```text
             precision    recall  f1-score   support

      B-art       0.37      0.11      0.17       402
      B-eve       0.52      0.35      0.42       308
      B-geo       0.85      0.90      0.88     37644
      B-gpe       0.97      0.94      0.95     15870
      B-nat       0.66      0.37      0.47       201
      B-org       0.78      0.72      0.75     20143
      B-per       0.84      0.81      0.82     16990
      B-tim       0.93      0.88      0.90     20333
      I-art       0.11      0.03      0.04       297
      I-eve       0.34      0.21      0.26       253
      I-geo       0.82      0.79      0.80      7414
      I-gpe       0.92      0.55      0.69       198
      I-nat       0.61      0.27      0.38        51
      I-org       0.81      0.79      0.80     16784
      I-per       0.84      0.89      0.87     17251
      I-tim       0.83      0.76      0.80      6528
          O       0.99      0.99      0.99    887908

    avg / total       0.97      0.97      0.97   1048575
  ```
- 04_Bi-LSTM

  句子长度统计：
  
  ![](https://github.com/yanqiangmiffy/ner-english/blob/master/assets/sentence_length.png)
  
  通过上图观察，句子最大长度max_len设置为50
  
  训练集和测试集：
  ```text
    X_train:(43163, 50)
    X_test(4796,50)
    y_train(43163,50,17)
    y_test(4796,50,17)
  ```
  **model:**
  
  ![](https://github.com/yanqiangmiffy/ner-english/blob/master/assets/BiLSTM.png)
  ```python
    input=Input(shape=(max_len,))
    model=Embedding(input_dim=n_words,output_dim=50,input_length=max_len)(input)
    model=Dropout(0.1)(model)
    model=Bidirectional(LSTM(units=100,return_sequences=True,recurrent_dropout=0.1))(model)
    out=TimeDistributed(Dense(n_tags,activation='softmax'))(model) # softmax output layer
    model=Model(input,out)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
  ```
  **训练结果:**
  ```text
    Epoch 1/5
    38846/38846 [==============================] - 90s 2ms/step - loss: 0.1410 - acc: 0.9643 - val_loss: 0.0622 - val_acc: 0.9818
    Epoch 2/5
    38846/38846 [==============================] - 88s 2ms/step - loss: 0.0550 - acc: 0.9838 - val_loss: 0.0517 - val_acc: 0.9849
    Epoch 3/5
    38846/38846 [==============================] - 88s 2ms/step - loss: 0.0459 - acc: 0.9865 - val_loss: 0.0477 - val_acc: 0.9860
    Epoch 4/5
    38846/38846 [==============================] - 89s 2ms/step - loss: 0.0413 - acc: 0.9878 - val_loss: 0.0459 - val_acc: 0.9865
    Epoch 5/5
    38846/38846 [==============================] - 89s 2ms/step - loss: 0.0385 - acc: 0.9885 - val_loss: 0.0444 - val_acc: 0.9868
  ```
  ![](https://github.com/yanqiangmiffy/ner-english/blob/master/assets/BiLSTM-result.png)
  
  **测试结果:**
  ```text
    Word           ||True ||Pred
    ==============================
    The            : O     O
    French         : B-gpe B-gpe
    news           : O     O
    agency         : O     O
    ,              : O     O
    Agence         : B-org O
    France         : I-org B-geo
    Presse         : I-org I-geo
    ,              : O     O
    says           : O     O
    one            : O     O
    of             : O     O
    its            : O     O
    photographers  : O     O
    has            : O     O
    been           : O     O
    kidnapped      : O     O
    in             : O     O
    the            : O     O
    Gaza           : B-geo B-geo
    Strip          : I-geo I-geo
    .              : O     O
  ```
- 05_Bi-LSTM+CRF
  
  **model:**
  
  ![](https://github.com/yanqiangmiffy/ner-english/blob/master/assets/BiLSTM-CRF.png)
  ```python
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)       
  ```
  
  **训练结果:**
  ```text
   Train on 38846 samples, validate on 4317 samples
   Epoch 1/5
   38846/38846 [==============================] - 137s 4ms/step - loss: 0.1651 - acc: 0.9546 - val_loss: 0.0691 - val_acc: 0.9766
   Epoch 2/5
   38846/38846 [==============================] - 136s 4ms/step - loss: 0.0513 - acc: 0.9815 - val_loss: 0.0429 - val_acc: 0.9834
   Epoch 3/5
   38846/38846 [==============================] - 131s 3ms/step - loss: 0.0365 - acc: 0.9855 - val_loss: 0.0376 - val_acc: 0.9849
   Epoch 4/5
   38846/38846 [==============================] - 132s 3ms/step - loss: 0.0315 - acc: 0.9871 - val_loss: 0.0344 - val_acc: 0.9859
   Epoch 5/5
   38846/38846 [==============================] - 131s 3ms/step - loss: 0.0287 - acc: 0.9879 - val_loss: 0.0339 - val_acc: 0.9857
  ```
  ![](https://github.com/yanqiangmiffy/ner-english/blob/master/assets/BiLSTM-CRF-result.png)

  **测试结果:**
  ```text
    Word           ||True ||Pred
    ==============================
    His            : O     O
    schedule       : O     O
    includes       : O     O
    talks          : O     O
    with           : O     O
    King           : B-per B-per
    Juan           : I-per I-per
    Carlos         : I-per I-per
    and            : O     O
    Spanish        : B-gpe B-gpe
    Prime          : B-per B-per
    Minister       : I-per I-per
    Jose           : I-per I-per
    Luis           : I-per I-per
    Rodriguez      : I-per I-per
    Zapatero       : I-per I-per
    .              : O     O
  ```
## 资料
https://www.one-tab.com/page/9-sFlWS0TTO_Kbcrnv4bqA
