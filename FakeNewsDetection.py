#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_csv("datasets/data.csv")
df = df.astype({"text": str})

def get_sentiment_score(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

def get_number_of_exclamation(text):
    count = text.count("!")
    return count

def get_number_of_Cap(text):
    return sum(1 for c in text if c.isupper())

def get_number_of_Space(text):
    return sum(1 for c in text if c==" ")

def num_Sentence(text):
    return sum(1 for c in text if c==".")

def num_num(text):
    sum = 0
    for i in range (len(text)-1):
        if text[i].isdigit() and (not text[i+1].isdigit()):
            sum+=1
    return sum

def have_author(text):
    if text != text:
        return 0
    else:
        return 1

df["Score"] = df['text'].apply(get_sentiment_score)
df["!"] = df['text'].apply(get_number_of_exclamation)
df["Upper"] = df['text'].apply(get_number_of_Cap)
df["Space"] = df['text'].apply(get_number_of_Space)
df["num_sentence"] = df['text'].apply(num_Sentence)
df["num_num"] = df['text'].apply(num_num)
df["have_author"] = df['author'].apply(have_author)

data = df.loc[:,['Score','!','Upper','Space','num_sentence','num_num','have_author']]
labels = df.loc[:,['label']]

# k range 
k_value_range = range(1,30)
# reult score
k_value_scores = []
K = 0
for k in k_value_range:
    knn_model = KNeighborsClassifier(n_neighbors = k)
    accuracy = cross_val_score(knn_model, data, labels.values.ravel(), cv=10, scoring="accuracy")
    #print("K:", k)
    #print("Accuracy: ", accuracy.mean())
    k_value_scores.append(accuracy.mean())
    #print(k_value_scores)
K = k_value_scores.index(max(k_value_scores))+1
print("Best K: " ,K)

plt.plot(k_value_range, k_value_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=0.25, random_state=1)

neigh = KNeighborsClassifier(n_neighbors=K)  
neigh.fit(X_train, y_train)
neigh.predict(X_test)

y_expect = y_test
y_predict = neigh.predict(X_test)
print("Accuracy = ",(y_expect==y_predict).mean())

print(metrics.classification_report(y_expect,y_predict))





