# CRF
import argparse
from utils import load_data,SentenceGetter,sent2features,sent2labels
from sklearn_crfsuite import CRF
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

# 1 加载数据
ner_dataset_dir='../data/ner_dataset.csv'
data=load_data(ner_dataset_dir)

# 2 构建数据集
getter=SentenceGetter(data)
sentences=getter.sentences
X=[sent2features(s) for s in sentences]
y=[sent2labels(s) for s in sentences]


# 3 CRF 训练 2
def train():
    crf=CRF(algorithm='lbfgs',
            c1=10,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True)

    pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
    report = flat_classification_report(y_pred=pred, y_true=y)
    print(report)

    crf.fit(X,y)
    # 保存模型
    joblib.dump(crf, "../result/crf.pkl")

def sample():
    crf=joblib.load(filename="../result/crf.pkl")
    pred=crf.predict([X[8]])[0]


    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w,p in zip(sentences[8],pred):
        print("{:15}: {:5} {}".format(w[0], w[2],p))




if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="命名执行训练或者预测")
    parser.add_argument('--action',required=True,help="input train or test")
    args=parser.parse_args()
    if args.action=='train':
        train()
    if args.action=='test':
        sample()