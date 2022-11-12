import fasttext
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

#https://huggingface.co/datasets/glue
#load data
import pandas as pd

df_train = load_dataset('glue', 'mrpc', split='train') 
df_test = load_dataset('glue', 'mrpc', split='validation') 

#data preparation for fasttext
with open('../../datasets_fasttext/mrpc.train', 'w', encoding="utf-8") as f:
    for each_premise, each_hypothesis, each_label in zip(df_train['sentence1'], df_train['sentence2'], df_train['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_premise} ñ {each_hypothesis}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/mrpc.valid', 'w', encoding="utf-8") as f:
    for each_premise, each_hypothesis, each_label in zip(df_test['sentence1'], df_test['sentence2'], df_test['label']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_premise} ñ {each_hypothesis}\n')



