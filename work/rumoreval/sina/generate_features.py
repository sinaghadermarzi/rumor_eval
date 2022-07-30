import csv
import sys
import json

def tweet_length(text):
    return (len(text))
def num_qmarks(text):
    return (text.count('?'))
def num_urls(text):
    return text.count("http://")

features_to_compute =[
    tweet_length,
    num_urls,
    num_qmarks,

]


from os import walk

filelist = []
for (dirpath, dirnames, filenames) in walk('parsed_files'):
    filelist.extend(filenames)
    break

outputfilename = "features.csv"
out_csv = open(outputfilename,"w",newline = '')

nn = 0
for fname in filelist:
    if fname.endswith(".json"):
        input_data_file_name = 'parsed_files/'+fname
        with open(input_data_file_name) as f:
            data = json.load(f)
            field_names = ["ID","Label"]
            for f in features_to_compute:
                field_names = field_names + [f.__name__]
            writer = csv.DictWriter(out_csv, field_names)
            if nn==0:
                writer.writeheader()
            kk = list(data.keys())
            main = data[kk[0]]
            a = dict()
            for post in main:
                # if True:#"#"replies" in post.keys():
                for p in post["replies"]:
                    # print(data)
                    #   loaded_replies = p["replies"]
                    #    print(loaded_replies)
                    ID = p["id_str"]
                    text = p["text"]
                    label = p["label"]
                    out_row = {}
                    out_row["ID"] = ID
                    for f in features_to_compute:
                        feature = f(text)
                        out_row[f.__name__] = str(feature)
                    out_row["Label"] = label
                    writer.writerow(out_row)
    nn+=1




