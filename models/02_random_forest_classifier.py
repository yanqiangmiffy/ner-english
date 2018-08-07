# 提取word特征，然后随机森林分类
import numpy as np
from utils import load_data
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
data=load_data(ner_dataset_dir)

def feature_map(word):
    """
    简单的特征：首字母是否大写，是否小写，是否为大写，单词长度，是否为数字，是否全为字母

    :return:
    """
    return np.array([word.istitle(),word.islower(),word.isupper(),len(word),
                     word.isdigit(),word.isalpha()])

# 2 简单测试下
words=[feature_map(word) for word in data['Word'].tolist()]
tags=data["Tag"].tolist()
pred=cross_val_predict(RandomForestClassifier(n_estimators=20),
                       X=words,y=tags,cv=5)

report=classification_report(y_pred=pred,y_true=tags)
print(report)

# 3 加强特征
from sklearn.base import BaseEstimator,TransformerMixin
class MemoryTagger(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        """
        words为X，tags为y
        :param X:
        :param y:
        :return:
        """
        voc={}
        self.tags=[]
        for x,t in zip(X,y):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t]+=1
                else:
                    voc[x][t]=1
            else:
                voc[x]={t:1}
        self.memory={}
        for k,d in voc.items():
            self.memory[k]=max(d,key=d.get) #  获取出现次数最多的标签
    def predict(self,X,y=None):
        """
        从memory预测tag，如果word是未知的，则预测为'O'
        :param X:
        :param y:
        :return:
        """
        return [self.memory.get(x,'O') for x in X]

from sklearn.preprocessing import LabelEncoder
class FeatureTransformer(BaseEstimator,TransformerMixin):

    def __init__(self):
        self.memory_tagger=MemoryTagger()
        self.tag_encoder=LabelEncoder()
        self.pos_encoder=LabelEncoder()

    def fit(self,X,y):
        words=X['Word'].tolist()
        self.pos=X['POS'].tolist() # 词性
        tags=X['Tag'].tolist()
        self.memory_tagger.fit(words,tags)
        self.tag_encoder.fit(tags)
        self.pos_encoder.fit(self.pos)
        return self

    def transform(self,X,y=None):
        def pos_default(p):
            if p in self.pos:
                return self.pos_encoder.transform([p])[0]
            else:
                return -1

        pos=X['POS'].tolist()
        words=X['Word'].tolist()
        out=[]

        for i in range(len(words)):
            w=words[i]
            p=pos[i]
            # 将上下文的信息考虑到特征之中
            if i<len(words)-1:
                wp=self.tag_encoder.transform(self.memory_tagger.predict([words[i]]))[0]
                posp=pos_default(pos[i+1])
            else:
                wp=self.tag_encoder.transform(['O'])[0]
                posp=pos_default('.')

            if i>0:
                if words[i-1]!='.':
                    wm=self.tag_encoder.transform(self.memory_tagger.predict(words[i-1]))[0]
                    posm=pos_default(pos[i-1])
                else:
                    wm=self.tag_encoder.transform(['O'])[0]
                    posm=pos_default('.')
            else:
                wm = self.tag_encoder.transform(['O'])[0]
                posm = pos_default(".")
            out.append(np.array([w.istitle(),w.islower(),w.isupper(),len(w),w.isdigit(),w.isalpha(),
                                 self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
                                 pos_default(p),wp,wm,posp,posm]))
        return out

# 4 训练预测
from sklearn.pipeline import Pipeline

pred=cross_val_predict(Pipeline([("feature_map",FeatureTransformer()),
                                 ("clf",RandomForestClassifier(n_estimators=20,n_jobs=3))]),
                       X=data,y=tags,cv=5)
report=classification_report(y_pred=pred,y_true=tags)
print(report)