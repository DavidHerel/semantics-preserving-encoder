import fasttext
from datasets import load_dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split

#https://huggingface.co/datasets/snli
#load data
import pandas as pd

df_train = load_dataset('stsb_multi_mt','en',split='train') 
df_test = load_dataset('stsb_multi_mt','en',split='test') 


df_tr = list(map(int, np.round(df_train['similarity_score'])))
df_te = list(map(int, np.round(df_test['similarity_score'])))
#data preparation for fasttext
with open('../../datasets_fasttext/stsb.train', 'w', encoding="utf-8") as f:
    for each_premise, each_hypothesis, each_label in zip(df_train['sentence1'], df_train['sentence2'], df_tr):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_premise} {each_hypothesis}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/stsb.valid', 'w', encoding="utf-8") as f:
    for each_premise, each_hypothesis, each_label in zip(df_test['sentence1'], df_test['sentence2'], df_te):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_premise} {each_hypothesis}\n')



