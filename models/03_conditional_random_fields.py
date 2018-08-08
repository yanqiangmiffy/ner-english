# CRF
import eli5
from models.utils import load_data,SentenceGetter
from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
data=load_data(ner_dataset_dir)

# 2 构建数据集
getter=SentenceGetter(data)
sent=getter.get_next()
# print(sent)

sentences=getter.sentences
# print(sentences[1:10])

# 2 提取特征
def word2features(sent,i):
    word=sent[i][0]
    postag=sent[i][1]

    features={
        'bias':1.0,
        'word.lower()':word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()':word.isupper(),
        'word.istitle()':word.istitle(),
        'word.isdigit()':word.isdigit(),
        'postag':postag,
        'postag[:2]':postag[:2]
    }

    if i>0:
        word1=sent[i-1][0]
        postag1=sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS']=True

    if i<len(sent)-1:
        word1=sent[i+1][0]
        postag1=sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS']=True

    return features

def sent2features(sent):
    return [word2features(sent,i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token,postag,label in sent]

def sent2tokens(sent):
    return [token for token,postag,label in sent]

X=[sent2features(s) for s in sentences]
y=[sent2labels(s) for s in sentences]


# 3 CRF 训练 1
crf=CRF(algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False)


pred=cross_val_predict(estimator=crf,X=X,y=y,cv=5)

report=flat_classification_report(y_pred=pred,y_true=y)
print(report)

crf.fit(X,y)
eli5.show_weights(crf,top=30)


# 3 CRF 训练 2
crf=CRF(algorithm='lbfgs',
        c1=10,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False)
pred=cross_val_predict(estimator=crf,X=X,y=y,cv=5)
report=flat_classification_report(y_pred=pred,y_true=y)
print(report)

crf.fit(X,y)
eli5.show_weights(crf,top=30)
