import fasttext
import os
from sklearn.model_selection import train_test_split
import pickle

## helper function
def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))

#https://huggingface.co/datasets/emotion
#load data
import pandas as pd

df = load_from_pickle(directory="../emotion_dataset.pkl")

# split the data into train and test set
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

#data preparation for fasttext
with open('../../datasets_fasttext/emotion.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(train['text'], train['emotions']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/emotion.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(test['text'], test['emotions']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')


