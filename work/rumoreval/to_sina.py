#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


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
        # print (reply_labels)

    # Data Cleaning
    ps = PorterStemmer()
    for oneReply in repliesText:
        cleanText = re.sub('http(\S)+', ' http ', oneReply)
        cleanText = re.sub('@', ' @ ', cleanText)
        cleanText = re.sub('#', ' # ', cleanText)
        cleanText = re.sub('\?', ' ? ', cleanText)
#         cleanText = re.sub('.', ' . ', cleanText)
        cleanText = re.sub('[0-9]+', ' number ', cleanText)
        
# #         cleanText = re.sub('@(\S)+', ' ', oneReply)
#         cleanText = re.sub('http(\S)+', 'http', oneReply)
# #         print('before', cleanText)
#         cleanText = re.sub('[^a-zA-Z\?]', ' ', cleanText) #TODO - What is this? Check with individual tweets
# #         print('after', cleanText)
#         cleanText = cleanText.lower()

        splitCleanText = cleanText.split()
#         splitCleanText = [word for word in splitCleanText if not word in set(stopwords.words('english'))] # TODO

        str = ""
        for oneWord in splitCleanText:
#             oneWord = ps.stem(oneWord) # TODO - stemming is a too aggressive here
            str += (' '+ oneWord)
            # str = ' '.join(oneWord)

        corpus.append(str)
    #print(corpus)

prepare_data(filenames)


# In[4]:


corpus


# In[5]:


# print(corpus)


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# cv = TfidfVectorizer(analyzer='word', ngram_range=(1,4), min_df = 0, stop_words = 'english', max_features= 500)
cv = TfidfVectorizer(analyzer='word', ngram_range=(1,4), min_df = 1, max_features= 300, norm='l2')

# TODO - this uses English stopwords already
# TODO - ngram_range (1, 3) 1, 2 and 3 grams are used.
# TODO - min_df => Ignore terms that have document frequency lower that 0.
# TODO - max_features =>  build a vocabulary that only consider the top max_features 
#                         ordered by term frequency across the corpus.

X = cv.fit_transform(corpus).toarray()
# TODO - By this point the tf-idf vectorization is already complete. Shouldn't we do this after SMOTE?
y = reply_labels


from sklearn.decomposition import PCA
n_components = 50
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)

print(X.shape)
X = pca.transform(X)
print(X.shape)
print(X[0])
# from sklearn.decomposition import TruncatedSVD

# print(X.shape, X[0])
# svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
# svd.fit(X) 
# print(X.shape, X[0])


# In[9]:


from collections import Counter
from imblearn.over_sampling import SMOTE # NOTE conda install -c glemaitre imbalanced-learn

# TODO - do not do SMOTE on the test/validation sets
print("Before SMOTE")
print(sorted(Counter(y).items()))

from collections import Counter
from imblearn.over_sampling import SMOTE
smote_enn = SMOTE(random_state=2)
X_resampled, y_resampled = smote_enn.fit_sample(X, y) # TODO - fit_resample(X,y) was not working.

print("AFTER SMOTE")
print(sorted(Counter(y_resampled).items()))


# In[11]:



# step : cross-validation
from sklearn.model_selection import train_test_split
import sklearn
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.40, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
# from imblearn.over_sampling import ADASYN

# from imblearn.over_sampling import BorderlineSMOTE # Not available in this version I guess
# from imblearn.over_sampling import SMOTENC
# smote_enn = SMOTE(ratio={'deny': 2000, 'query': 2000, 'support': 2000}, random_state=2, kind='regular', svm_estimator=sklearn.svm.SVC, out_step=0.1)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# test = SelectKBest(score_func=chi2, k=10)
# fit = test.fit(X_train, y_train)
# X_train = fit.transform(X_train)


# approach1:tree
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)

best_features  = np.argsort(model.feature_importances_)
num_feature = 50
X= X[:,best_features[-num_feature:]]




smote_enn = SMOTE(ratio={'deny': 1500, 'query': 1500, 'support': 1500})
smote_enn = SMOTE()
X_smote, y_smote = smote_enn.fit_sample(X_train, y_train)



#
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# test = SelectKBest(score_func=chi2, k=10)
# fit = test.fit(X_smote, y_smote)
# X_smote = fit.transform(X_smote)



#step: Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_smote, y_smote)
# clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

print(sorted(Counter(y_test).items()))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)
#print(len(reply_labels)
recall = np.diag(cm) / np.sum(cm, axis=1)
precision = np.diag(cm)/ np.sum(cm, axis=0)

f1_score = 2* (precision*recall)/(precision + recall + 1)
acc = accuracy_score(y_test, y_predict)
print('precision = ',precision)
print('recall = ', recall)
print('f1 score = ',f1_score)
print('RF accuracy = ',acc)

print("\n")


# In[ ]:




