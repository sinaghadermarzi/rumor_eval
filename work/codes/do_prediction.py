
import pandas



#we will build a model with a selction of features and then output it in the form of a




selected_features = []


train_dataset = pandas.read_csv("")
train_labels = numpy.array(train_dataset['label'])
train_dataset= train_dataset.drop('label', axis = 1)
train_dataset= train_dataset.drop('ID', axis = 1)
train_features = numpy.array(train_dataset)



test_dataset = pandas.read_csv("")
test_labels = numpy.array(test_dataset['label'])
test_dataset= test_dataset.drop('label', axis = 1)
test_dataset= test_dataset.drop('ID', axis = 1)
test_features = numpy.array(test_dataset)



















set_num_trees = [8,16,32,64]
set_max_depth = [2,4,8,16,32]
best_res  = 0
best_max_depth = 0
best_num_trees = 0
#build a random forest with a setting of hyperparameters

for num_trees in set_num_trees:
    for max_depth in set_max_depth:
        for i in range(0,num_folds):
            train_fold = []
            for j in range(0, num_training_objs):
                if j not in list_test_fold_idx[i]:
                    train_fold.append(j)
            list_training_fold_idx.append(train_fold)
        aucs= []
        for i in range(0, num_folds):
            train_fold_features= train_features[list_training_fold_idx[i],:]
            train_fold_labels=train_labels[list_training_fold_idx[i]]
            test_fold_features= test_features[list_training_fold_idx[i],:]
            train_fold_labels = test_labels[list_training_fold_idx[i]]
            clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=num_trees,max_depth=max_depth)
            clf.fit(train_fold_features, train_fold_labels)
            preds = clf.predict(test_fold_features)
            scores = clf.predict_proba(test_features)
            auc_query = roc_auc_score(test_labels == "query", scores[:, 0])
            auc_comment = roc_auc_score(test_labels == "comment", scores[:, 1])
            auc_deny = roc_auc_score(test_labels == "deny", scores[:, 0])
            auc_support = roc_auc_score(test_labels == "support", scores[:, 1])
            auc_average = numpy.mean([auc_comment,auc_support,auc_deny,auc_query])
            aucs.append(auc_average)
        print('num_trees=',num_trees,"max_depth=", max_depth,":", aucs)
        res = numpy.mean(aucs)
        if res>best_res:
            best_num_trees = num_trees
            best_max_depth = max_depth
            best_res = res




clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=best_num_trees,max_depth=best_max_depth)
clf.fit(train_features,train_labels)
preds = clf.predict(test_features)
# print("\n\n\npredictions")
# print(preds)
scores = clf.predict_proba(test_features)
confusion_matrix = pandas.crosstab(test_labels, preds, rownames=['Actual Resolution'], colnames=['Predicted Resolution'])
print("\n\n\nconfusion matrix")
print(confusion_matrix)
auc_good = roc_auc_score(test_labels=="good",scores[:,0])
auc_poor = roc_auc_score(test_labels=="poor",scores[:,1])
print ("auc_good",auc_good)
print ("auc_poor",auc_poor)

