import pandas
import json

IDs = []
preds = []
true_values = []


#imports a csv file as predictions and calculates the confusion matrix and other measures
prediction_file_address = 'predictions.csv'
true_values_file_address = 'dev-key.json'
with open(prediction_file_address) as prediction_file, open(true_values_file_address) as true_values_file:
    data = json.load(true_values_file)
    #read all true values into a dict
    true_value = data["subtaska"]
    #read all prediction values into a dict
    line = prediction_file.readline()
    prediction = {}
    while(line):
        line = line.rstrip('\n')
        line_spl = line.split(',')
        id  = line_spl[0]
        id  = id[1:len(id)-1]
        pr = line_spl[1]
        pr = pr[1:len(pr)-1]
        prediction[id] = pr
        line = prediction_file.readline()
    for id in prediction.keys():
        pr = prediction[id]
        preds.append(pr)
        true_values.append(true_value[id])
    confusion_matrix = pandas.crosstab([true_values], [preds],rownames = ["true values"], colnames=["predictions"])#, rownames=['True Label'] , colnames = ['Predicted Label'])
    print("\n\n\nconfusion matrix")
    print(confusion_matrix)
    # auc_good = roc_auc_score(test_labels == "good", scores[:, 0])
    # auc_poor = roc_auc_score(test_labels == "poor", scores[:, 1])
    # print ("auc_good", auc_good)
    # print ("auc_poor", auc_poor)
