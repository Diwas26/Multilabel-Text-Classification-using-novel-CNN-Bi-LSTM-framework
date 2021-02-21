# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:08:12 2021

@author: rhapsody
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

df = pd.read_csv(r"Dataset_toxic_comment\train.csv", encoding='iso-8859-1')
df.info() ## info of the dataframe ##
df.head(n=5) ## top-5 values of the dataframe ##

## Taking sample to work on just for framework demostration ##

df_data = df.sample(frac=0.2, replace=True, random_state=1)
## Check for Null Values ##
df_data.isnull().sum()


###############################################################################
############################ Word Pre-Processing ##############################

import nltk
import string
wpt = nltk.WordPunctTokenizer()
stop_words_init = nltk.corpus.stopwords.words('english')
stop_words = [i for i in stop_words_init if i not in ('not','and','for')]
print(stop_words)

## Function to normalize text for pre-processing ##
def normalize_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    return text

## Apply the written function ##
df_data['comment_text'] = df_data['comment_text'].apply(lambda x: normalize_text(x))

df_data['comment_text'].head(n=5)

processed_list = []
for j in df_data['comment_text']:
    process = j.replace('...','')
    processed_list.append(process)
    
df_processed = pd.DataFrame(processed_list)
df_processed.columns = ['comments']
df_processed.head(n=5)


############# Now, checking the label's availabel per classes #################

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = df_data[labels].values

import matplotlib.pyplot as plt

val_counts = df_data[labels].sum()

plt.figure(figsize=(8,5))
ax = sns.barplot(val_counts.index, val_counts.values, alpha=0.8)

plt.title("Labels per Classes")
plt.xlabel("Various Label Type")
plt.ylabel("Counts of the Labels")

rects = ax.patches
labels = val_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha="center", va="bottom")

plt.show()


############ Plotting for Data Imbalance ######################################
import seaborn as sns

fig , axes = plt.subplots(2,3,figsize = (10,10), constrained_layout = True)
sns.countplot(ax=axes[0,0],x='toxic',data=df_data )
sns.countplot(ax=axes[0,1],x='severe_toxic',data=df_data)
sns.countplot(ax=axes[0,2],x='obscene',data=df_data)
sns.countplot(ax = axes[1,0],x='threat',data=df_data)
sns.countplot(ax=axes[1,1],x='insult',data=df_data)
sns.countplot(ax=axes[1,2],x='identity_hate',data=df_data)
plt.suptitle('Number Of Labels of each Toxicity Type')
plt.show()


#################### Preparing data for training and validation ###############

X = list(df_processed['comments'])
y_data = df_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
y = y_data.values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.15, train_size=0.85)
