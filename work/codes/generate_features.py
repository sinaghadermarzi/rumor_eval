import csv
import sys
import json

def tweet_length(text):
    return (len(text))

features_to_compute = [
    tweet_length
]


from os import walk

filelist = []
for (dirpath, dirnames, filenames) in walk('parsed_files'):
    filelist.extend(filenames)
    break



input_data_file_name = 'charliehebdo.json'




with open(input_data_file_name) as f,open(input_data_file_name[:-4] + '_features.csv','w',newline='') as out_csv:
    data = json.load(f)
    field_names = ["ID"]
    for f in features_to_compute:
        field_names = field_names + [f.__name__]
    writer = csv.DictWriter(out_csv, field_names)
    writer.writeheader()
    for post in data["charliehebdo"]:
        for p in post["replies"]:
            # print(data)
            #   loaded_replies = p["replies"]
            #    print(loaded_replies)
            ID = p["id_str"]
            text = p["text"]
            out_row = {}
            out_row["ID"] = ID
            for f in features_to_compute:
                feature = f(text)
                out_row[f.__name__] = str(feature)
            writer.writerow(out_row)





