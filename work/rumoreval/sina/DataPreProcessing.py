# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 17:49:28 2018

@author: palak
"""
import json


with open('parsed_files_charliehebdo.json') as f:
    data = json.load(f)
#pprint(data)
with open("tabular_dataset.csv","w", newline = '') as outfile:
    outfile.writelines('id,text\n')
    for post in data["charliehebdo"]:
        for p in post["replies"]:
            
            #print(data)
         #   loaded_replies = p["replies"]
        #    print(loaded_replies)
            ID=p["id_str"]
            text = p["text"]
           # print(ID)
            #print (text)
            print(ID+','+text+'\n')
            outfile.writelines(ID+","+text+'\n')