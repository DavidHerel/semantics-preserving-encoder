import fasttext
import os
from sklearn.model_selection import train_test_split

#https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
#load data
import pandas as pd
df = pd.read_json("../Sarcasm_Headlines_Dataset_v2.json", lines=True)
#shuffle the data inplace
df = df.sample(frac=1).reset_index(drop=True)

# split the data into train and test set
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

#data preparation for fasttext
with open('../../datasets_fasttext/sarcastic.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(train['headline'], train['is_sarcastic']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/sarcastic.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(test['headline'], test['is_sarcastic']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

