import fasttext
import os
from sklearn.model_selection import train_test_split

#https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews?select=tripadvisor_hotel_reviews.csv
#load data
import pandas as pd
df = pd.read_csv("../tripadvisor_hotel_reviews.csv")

train, valid = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

#data preparation for fasttext
with open('../../datasets_fasttext/trip_advisor.train', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(train['Review'], train['Rating']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')

#data preparation for fasttext
with open('../../datasets_fasttext/trip_advisor.valid', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(valid['Review'], valid['Rating']):
        #print(f'__label__{each_label} {each_text}\n')
        f.writelines(f'__label__{each_label} {each_text}\n')


