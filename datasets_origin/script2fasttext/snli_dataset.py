import fasttext
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

#https://huggingface.co/datasets/snli
#load data
import pandas as pd

df_train = load_dataset('snli',split='train') 
df_test = load_dataset('snli',split='test') 

#data preparation for fasttext
with open('../../datasets_fasttext/snli.train', 'w', encoding="utf-8") as f:
    for each_premise, each_hypothesis, each_label in zip(df_train['premise'], df_train['hypothesis'], df_train['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_premise} {each_hypothesis}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/snli.valid', 'w', encoding="utf-8") as f:
    for each_premise, each_hypothesis, each_label in zip(df_test['premise'], df_test['hypothesis'], df_test['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_premise} {each_hypothesis}\n')



