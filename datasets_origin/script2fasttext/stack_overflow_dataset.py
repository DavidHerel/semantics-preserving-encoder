import fasttext
import os
from sklearn.model_selection import train_test_split
import re

#https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate
#load data

import pandas as pd
df_train = pd.read_csv("../stack_overflow_train.csv")
df_valid = pd.read_csv("../stack_overflow_valid.csv")

df_train['Body'] = df_train['Title'] + " " + df_train['Body']
df_valid['Body'] = df_valid['Title'] + " " + df_valid['Body']

# Clean the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)\s]','', text)
    text = text.replace('\n','')
    return text

df_train['Body'] = df_train['Body'].apply(clean_text)
df_valid['Body'] = df_valid['Body'].apply(clean_text)

#data preparation for fasttext
with open('../../datasets_fasttext/stack_overflow.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_train['Body'], df_train['Y']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/stack_overflow.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_valid['Body'], df_valid['Y']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')


