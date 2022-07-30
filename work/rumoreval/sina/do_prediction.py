from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import *


#we will build a model with a selction of features and then output it in the form of a



selected_features = []


train_dataset = pandas.read_csv("features.csv")
train_labels = numpy.array(train_dataset['Label'])
train_IDs = numpy.array(train_dataset['ID'])
train_dataset= train_dataset.drop('Label', axis = 1)
train_dataset= train_dataset.drop('ID', axis = 1)
train_features = numpy.array(train_dataset)



test_dataset = pandas.read_csv("features_test.csv")
test_labels = numpy.array(test_dataset['Label'])
test_IDs = numpy.array(test_dataset['ID'])
test_dataset= test_dataset.drop('Label', axis = 1)
test_dataset= test_dataset.drop('ID', axis = 1)
test_features = numpy.array(test_dataset)






num_training_objs, num_features= train_dataset.shape

# train_fold_features = train_features[list_training_fold_idx[i], :]
# train_fold_labels = train_labels[list_training_fold_idx[i]]
# test_fold_features = test_features[list_training_fold_idx[i], :]
# train_fold_labels = test_labels[list_training_fold_idx[i]]
# clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=num_trees, max_depth=max_depth)
# clf.fit(train_fold_features, train_fold_labels)
# preds = clf.predict(test_fold_features)
# scores = clf.predict_proba(test_features)
# auc_query = roc_auc_score(test_labels == "query", scores[:, 0])
# auc_comment = roc_auc_score(test_labels == "comment", scores[:, 1])
# auc_deny = roc_auc_score(test_labels == "deny", scores[:, 0])
# auc_support = roc_auc_score(test_labels == "support", scores[:, 1])
# auc_average = numpy.mean([auc_comment, auc_support, auc_deny, auc_query])
# aucs.append(auc_average)












num_trees = 4
max_depth = 4




# num_folds = 3
# num_training_objs = len(train_labels)
# num_test_objs = len(train_labels)
# shuffled_idx = list(range(0, num_training_objs))
# numpy.random.shuffle(shuffled_idx)
#
# fold_boundary = []
# list_test_fold_idx = []
# for i in range(1, num_folds):
#     test_fold_start = int((i - 1) * numpy.floor(float(num_training_objs) / num_folds))
#     test_fold_end = int(i * numpy.floor(float(num_training_objs) / num_folds))
#     test_fold_idx = tuple(shuffled_idx[test_fold_start:test_fold_end])
#     list_test_fold_idx.append(test_fold_idx)
# test_fold_start = int((num_folds - 1) * numpy.floor(float(num_training_objs) / num_folds))
# test_fold_idx = tuple(shuffled_idx[test_fold_start:num_training_objs])
# list_test_fold_idx.append(test_fold_idx)
#
# list_training_fold_idx = []


# clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=num_trees, max_depth=max_depth)

clf = SVC()
clf.fit(train_features, train_labels)
preds = clf.predict(test_features)

preds_majority = []
for i in range(0,len(preds)):
    preds_majority.append( "comment")
preds_majority = numpy.array(preds_majority)

# scores = clf.predict_proba(test_features)




# auc_query = roc_auc_score(test_labels == "query", scores[:, 0])
# auc_comment = roc_auc_score(test_labels == "comment", scores[:, 1])
# auc_deny = roc_auc_score(test_labels == "deny", scores[:, 0])
# auc_support = roc_auc_score(test_labels == "support", scores[:, 1])
# auc_average = numpy.mean([auc_comment, auc_support, auc_deny, auc_query])

acc = accuracy_score(test_labels,preds)
outfile = open("prediction_results.csv", "w", newline='')
for i in range(0,len(test_IDs)):
    outfile.writelines("\""+str(test_IDs[i])+"\""+','+"\""+preds[i]+"\""+'\n')
outfile.close()
print("accuracy = " + str(acc))
confusion_matrix = pandas.crosstab(test_labels, preds, rownames=['Actual Resolution'], colnames=['Predicted Resolution'])
print("\n\n\nconfusion matrix")
print(confusion_matrix)
f1 = f1_score(test_labels,preds,average='macro' )
print("F1 = " + str(f1))

acc = accuracy_score(test_labels,preds_majority)
print("accuracy of majority= " + str(acc))
# f1 = f1_score(test_labels,preds,average='macro' )
# print("F1 = " + str(f1))
confusion_matrix = pandas.crosstab(test_labels, preds_majority, rownames=['Actual Resolution'], colnames=['Predicted Resolution'])
print("\n\n\nconfusion matrix of majority")
print(confusion_matrix)



# set_num_trees = [8,16,32,64]
# set_max_depth = [2,4,8,16,32]
# best_res  = 0
# best_max_depth = 0
# best_num_trees = 0
# #build a random forest with a setting of hyperparameters
#
# for num_trees in set_num_trees:
#     for max_depth in set_max_depth:
#         for i in range(0,num_folds):
#             train_fold = []
#             for j in range(0, num_training_objs):
#                 if j not in list_test_fold_idx[i]:
#                     train_fold.append(j)
#             list_training_fold_idx.append(train_fold)
#         aucs= []
#         for i in range(0, num_folds):
#             train_fold_features= train_features[list_training_fold_idx[i],:]
#             train_fold_labels=train_labels[list_training_fold_idx[i]]
#             test_fold_features= test_features[list_training_fold_idx[i],:]
#             train_fold_labels = test_labels[list_training_fold_idx[i]]
#             clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=num_trees,max_depth=max_depth)
#             clf.fit(train_fold_features, train_fold_labels)
#             preds = clf.predict(test_fold_features)
#             scores = clf.predict_proba(test_features)
#             auc_query = roc_auc_score(test_labels == "query", scores[:, 0])
#             auc_comment = roc_auc_score(test_labels == "comment", scores[:, 1])
#             auc_deny = roc_auc_score(test_labels == "deny", scores[:, 0])
#             auc_support = roc_auc_score(test_labels == "support", scores[:, 1])
#             auc_average = numpy.mean([auc_comment,auc_support,auc_deny,auc_query])
#             aucs.append(auc_average)
#         print('num_trees=',num_trees,"max_depth=", max_depth,":", aucs)
#         res = numpy.mean(aucs)
#         if res>best_res:
#             best_num_trees = num_trees
#             best_max_depth = max_depth
#             best_res = res
#
#
#
#
# clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=best_num_trees,max_depth=best_max_depth)
# clf.fit(train_features,train_labels)
# preds = clf.predict(test_features)
# # print("\n\n\npredictions")
# # print(preds)
# scores = clf.predict_proba(test_features)
# confusion_matrix = pandas.crosstab(test_labels, preds, rownames=['Actual Resolution'], colnames=['Predicted Resolution'])
# print("\n\n\nconfusion matrix")
# print(confusion_matrix)
# auc_good = roc_auc_score(test_labels=="good",scores[:,0])
# auc_poor = roc_auc_score(test_labels=="poor",scores[:,1])
# print ("auc_good",auc_good)
# print ("auc_poor",auc_poor)

