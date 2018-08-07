# ner-english
英文命名实体识别(NER)的研究

## 准备
数据集：Kaggle-https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/4

实体标签含义：
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
- 02_random_forest_classifier:

  基本特征：`首字母是否大写，是否小写，是否为大写，单词长度，是否为数字，是否全为字母`
  
  上下文特征:`上下文单词的标签以及词性特征`
  
  方法：`RandomForestClassifier`

## 资料
https://www.one-tab.com/page/9-sFlWS0TTO_Kbcrnv4bqA
