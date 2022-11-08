import fasttext
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

#https://huggingface.co/datasets/trec
#load data
import pandas as pd

df_train = load_dataset('trec',split='train') 
df_test = load_dataset('trec',split='test') 

#data preparation for fasttext
with open('../../datasets_fasttext/trec.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_train['text'], df_train['label-coarse']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/trec.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_test['text'], df_test['label-coarse']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')



