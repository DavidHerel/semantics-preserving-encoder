import fasttext
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

#https://huggingface.co/datasets/imdb
#load data
import pandas as pd

df_train = load_dataset('imdb',split='train') 
df_test = load_dataset('imdb',split='test') 

#data preparation for fasttext
with open('../../datasets_fasttext/imdb.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_train['text'], df_train['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/imdb.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_test['text'], df_test['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')



