import fasttext
import os
from sklearn.model_selection import train_test_split

_LABEL_MAPPING = {-1: 0, 0: 2, 1: 1, 2: 3}

#https://huggingface.co/datasets/poem_sentiment
#load data
import pandas as pd
df_train = pd.read_csv("../poem_train.tsv",sep="\t")
df_valid = pd.read_csv("../poem_test.tsv",sep="\t")

#data preparation for fasttext
with open('../../datasets_fasttext/poem.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_train['text'], df_train['label']):
        #print(f'__label__{each_label} {each_text}\n')
        each_label = _LABEL_MAPPING[int(each_label)]
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/poem.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_valid['text'], df_valid['label']):
        #print(f'__label__{each_label} {each_text}\n')
        each_label = _LABEL_MAPPING[int(each_label)]
        f.writelines(f'__label__{each_label} {each_text}\n')



