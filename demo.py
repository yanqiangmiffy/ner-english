from models.utils import load_data,SentenceGetter
import random



# 1 加载数据
ner_dataset_dir='data/ner_dataset.csv'
dataset_dir='data/dataset.pkl'
data=load_data(ner_dataset_dir)

getter=SentenceGetter(data)
sentences=getter.sentences
sentences=[" ".join([w[0] for w in s ]) for s in sentences]
print(random.choice(sentences))