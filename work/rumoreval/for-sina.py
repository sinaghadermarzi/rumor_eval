#!/usr/bin/env python
# coding: utf-8

# In[118]:


import json, re
import nltk
import numpy as np
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

filenames = [
    'charliehebdo',
    'ferguson',
    'illary',
    'prince-toronto',
    'sydneysiege',
    'ebola-essien',
    'germanwings-crash',
    'ottawashooting',
    'putinmissing'
]
    


# In[125]:


embedding_dict = {}
n_features = 200

def prepare_embeddings():
    print("Loading Global Vectors....")
    if n_features == 100:
        embeddings_file = 'C:/Users/sinag/Documents/glove.twitter.27B/glove.twitter.27B.100d.txt'
    else:
        embeddings_file = 'C:/Users/sinag/Documents/glove.twitter.27B/glove.twitter.27B.200d.txt'

    f = []
    with open(embeddings_file,encoding="utf8") as file:
        f = file.readlines()

    for fi in f:
        line = fi.split()
        embedding_dict[line[0]] = line[1:] # TODO - convert into float here itself

prepare_embeddings()
print("Completed loading Global Vectors.")


# In[126]:




print(len(embedding_dict))
print(embedding_dict['for'])
print('is' in embedding_dict)


# In[127]:


def load_data_from_file(name):
    filename = 'parsed_files/' + name + '.json'
    # print(filename)

    try:
        with open(filename) as f:
            tweets = json.load(f)
    except Exception as e:
        print("Exception")
        print(e.message)

    return tweets


# In[191]:


repliesText = []
reply_labels = []
corpus = []

def prepare_data(filenames):
    for name in filenames:
        t = load_data_from_file(name)

        for i in range(len(t[name])):
            thread = t[name][i]
            reply_list = thread['replies']
            for reply in reply_list:
                text = reply["text"]
                label = reply["label"]
                repliesText.append(text)
                reply_labels.append(label)

    # Data Cleaning
    ps = PorterStemmer()
    for oneReply in repliesText:
        cleanText = re.sub('http(\S)+', ' http ', oneReply)
        cleanText = re.sub('@', '', cleanText)
        cleanText = re.sub('#', '', cleanText)
        cleanText = re.sub('\?', ' ? ', cleanText)
        cleanText = re.sub('\.', ' . ', cleanText)
        cleanText = re.sub('\!', ' ! ', cleanText)
        cleanText = re.sub('\_', ' _ ', cleanText)
        cleanText = re.sub("'", '', cleanText)
        cleanText = re.sub("'", " ' "  , cleanText)
        cleanText = re.sub(":", " : "  , cleanText)
        cleanText = re.sub(";", " ; "  , cleanText)
        cleanText = re.sub(",", " , "  , cleanText)
        cleanText = re.sub("-", " - "  , cleanText)
        cleanText = re.sub("\(", " ( "  , cleanText)
        cleanText = re.sub("\)", " ) "  , cleanText)
        cleanText = re.sub('[0-9]+', ' number ', cleanText)

        cleanText = cleanText.lower()

        splitCleanText = cleanText.split()

        str = ""
        for oneWord in splitCleanText:
#             oneWord = ps.stem(oneWord) # TODO - stemming is a too aggressive here
            str += (' '+ oneWord)
            # str = ' '.join(oneWord)

        corpus.append(str)

prepare_data(filenames)


# In[192]:


corpus


# In[194]:


# X = [[] for i in range(len(corpus))]
X = np.zeros((5046, n_features))
print(X.shape)
print(X[0].shape)
y = reply_labels
# n_features = 100

a = np.array([float(s) for s in embedding_dict['for']]).reshape(n_features, 1)

print(len(corpus))
for i in range(len(corpus)):
    sentence = corpus[i]
    # for sentence in corpus:
    words = sentence.split()
    x_ = np.zeros((n_features, 1))
    l = len(words)
    for word in words:
        if word in embedding_dict:
            feat = np.array([float(s) for s in embedding_dict[word]]).reshape(n_features, 1)
            x_ = np.add(x_, feat)
#             print(feat.shape)
        else:
            print("word not found", word) # NOTHING YES!!
            if word == "don’t":
                print(sentence)
#     x_ = np.divide(x_, l).reshape(n_features,)
    x_ = x_.reshape(n_features,)
    X[i] = x_


# In[195]:


# X = X.reshape(5046, n_features)
# print(X.shape)
# print(y.shape)

# a = np.zeros((3, 2))
# b = np.array([1, 3])
# print(a)
# print(b)
# a[1] = b
# print(a)


# In[199]:


from sklearn.model_selection import train_test_split
import sklearn
from collections import Counter

from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
print(X_train.shape)
print(X_train[0].shape)

smote_enn = SMOTE(random_state=2)
X_resampled, y_resampled = smote_enn.fit_sample(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
num_trees = 4
max_depth = 4
clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=num_trees, max_depth=max_depth)
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)
clf.fit(X_resampled, y_resampled)


# In[200]:


y_predict = clf.predict(X_test)

print(sorted(Counter(y_test).items()))
print(sorted(Counter(y_predict).items()))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)
#print(len(reply_labels)
recall = np.diag(cm) / np.sum(cm, axis=1)
precision = np.diag(cm)/ np.sum(cm, axis=0)

f1_score = 2 * (precision*recall)/(precision + recall + 1)
acc = accuracy_score(y_test, y_predict)
print('precision = ',precision)
print('recall = ', recall)
print('f1 score = ',f1_score)
print('RF accuracy = ',acc)

print("\n")


# In[ ]:




