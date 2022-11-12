import fasttext
import os
from sklearn.model_selection import train_test_split

#https://huggingface.co/datasets/banking77
#load data
import pandas as pd
df_train = pd.read_csv("../banking_train.csv")
df_valid = pd.read_csv("../banking_test.csv")

#data preparation for fasttext
with open('../../datasets_fasttext/banking.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_train['text'], df_train['category']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/banking.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_valid['text'], df_valid['category']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')




