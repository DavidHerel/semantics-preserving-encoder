import fasttext
import os
from sklearn.model_selection import train_test_split

#https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
#load data
import pandas as pd
df = pd.read_csv("../Womens Clothing E-Commerce Reviews.csv")
#shuffle the data inplace
df = df.sample(frac=1).reset_index(drop=True)

df = df[df["Rating"] != 3]
#print(df["Rating"] > 3)
df['Sentiment'] = df["Rating"] >3
df['Review Text'] = df['Review Text'].str.replace('\n','')
# split the data into train and test set
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

#data preparation for fasttext
with open('../../datasets_fasttext/clothing.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(train['Review Text'], train['Sentiment']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/clothing.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(test['Review Text'], test['Sentiment']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')
