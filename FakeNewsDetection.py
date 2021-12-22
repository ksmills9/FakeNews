#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[2]:


pd.set_option("display.max_columns", None)


# In[3]:


df = pd.read_csv("datasets/data.csv")
df = df.astype({"text": str})
df.head(5)


# In[4]:


df.info()


# In[5]:


def get_sentiment_score(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment


# In[6]:


def get_number_of_exclamation(text):
    countExclamation = text.count("!")
    countQuestion = text.count("?")
    return countExclamation + countQuestion


# In[7]:


def get_number_of_Cap(text):
    return sum(1 for c in text if c.isupper())


# In[8]:


import re
def number_of_all_cap_words(text):
    return len((re.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', text)))


# In[9]:


def num_Sentence(text):
    textSentences = re.split(r'[.!?]+', text)
    return len(textSentences)-1


# In[10]:


def num_num(text):
    sum = 0
    for i in range (len(text)-1):
        if text[i].isdigit() and (not text[i+1].isdigit()):
            sum+=1
    return sum


# In[11]:


def have_author(text):
    if text != text:
        return 0
    else:
        return 1


# In[12]:


df["Score"] = df['text'].apply(get_sentiment_score)


# In[13]:


df["?!"] = df['text'].apply(get_number_of_exclamation)


# In[14]:


df["Upper"] = df['text'].apply(get_number_of_Cap)


# In[15]:


df["num_sentence"] = df['text'].apply(num_Sentence)


# In[16]:


df["num_num"] = df['text'].apply(num_num)


# In[17]:


df["have_author"] = df['author'].apply(have_author)


# In[18]:


df["CAP"]=df['text'].apply(number_of_all_cap_words)


# In[19]:


df.head(5)


# In[20]:


data = df.loc[:,['Score','?!','Upper','num_sentence','num_num','have_author','CAP']]
labels = df.loc[:,['label']]


# In[21]:


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
print("Best K: " ,k_value_scores.index(max(k_value_scores))+1)

K = k_value_scores.index(max(k_value_scores))+1

plt.plot(k_value_range, k_value_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy')
plt.show()


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=0.25, random_state=1)


# In[23]:


X_train.head()


# In[24]:


X_test.head()


# In[25]:


print(y_train)


# In[26]:


print(y_test)


# In[27]:


neigh = KNeighborsClassifier(n_neighbors=K)
neigh.fit(X_train, y_train)


# In[28]:


neigh.predict(X_test)


# In[29]:


y_expect = y_test
y_predict = neigh.predict(X_test)
(y_expect==y_predict).mean()


# In[30]:


print(metrics.classification_report(y_expect,y_predict))
