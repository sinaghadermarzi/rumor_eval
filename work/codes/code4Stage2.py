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
        cleanText = re.sub('@(\S)+', ' ', oneReply)
        cleanText = re.sub('http(\S)+', ' ', cleanText)
        cleanText = re.sub('[^a-zA-Z]', ' ', cleanText)
        # print(cleanText)
        cleanText = cleanText.lower()

        splitCleanText = cleanText.split()
        splitCleanText = [word for word in splitCleanText if not word in set(stopwords.words('english'))]

        str = ""
        for oneWord in splitCleanText:
            oneWord = ps.stem(oneWord)
            str += (' '+ oneWord)
            # str = ' '.join(oneWord)

        corpus.append(str)
    #print(corpus)

prepare_data(filenames)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#cv = CountVectorizer(max_features=50000)
cv = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english', max_features= 150)
X = cv.fit_transform(corpus).toarray()
y = reply_labels



from collections import Counter
from sklearn.datasets import make_classification
X_train, y_train = make_classification(n_samples=5046, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=4,
                          n_clusters_per_class=1,
                           weights=[0.90, 0.01, 0.05, 0.04],
                           class_sep=0.8, random_state=0)
print(sorted(Counter(y).items()))

#from imblearn.under_sampling import ClusterCentroids
#cc = ClusterCentroids(random_state=0)

#from imblearn.under_sampling import RandomUnderSampler
#rus = RandomUnderSampler(random_state=0)

from collections import Counter
from imblearn.over_sampling import SMOTE
smote_enn = SMOTE(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

print(sorted(Counter(y_resampled).items()))


# step : cross-validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.40, random_state=0)

#step: Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_resampled, y_resampled)
y_predict = clf.predict(X_test)



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
