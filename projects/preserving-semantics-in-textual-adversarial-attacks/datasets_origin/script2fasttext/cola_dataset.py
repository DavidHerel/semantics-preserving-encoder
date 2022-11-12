import fasttext
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

#https://huggingface.co/datasets/glue
#load data
import pandas as pd

df_train = load_dataset('glue', 'cola', split='train') 
df_test = load_dataset('glue', 'cola', split='validation') 

#data preparation for fasttext
with open('../../datasets_fasttext/cola.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_train['sentence'], df_train['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/cola.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(df_test['sentence'], df_test['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')



