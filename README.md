# ner-english
:leopard: 英文命名实体识别(NER)的研究

## 准备
- 数据集：Kaggle-https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/4
- 词汇量：去重之后：`35178`
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

## 资料
https://www.one-tab.com/page/9-sFlWS0TTO_Kbcrnv4bqA
