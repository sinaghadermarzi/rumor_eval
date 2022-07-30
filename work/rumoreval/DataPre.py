# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:38:57 2018

@author: palak
"""

# Used for Extracting all source and reply tweets
from nltk.tokenize import TweetTokenizer
import json, os, re


maxLevel = []
maxLengthTweet = []

#..............................TRAIN..................................................
folder = r"C:/Users/palak/Desktop/CMSC516_NLP"
tknzr = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)

with open('C:/Users/palak/Desktop/CMSC516_NLP/rumoureval-2019-training-data/rumoureval-2019-training-data/train-key.json', 'r') as rlT:
    replyLabelTrain = json.load(rlT)
with open('C:/Users/palak/Desktop/CMSC516_NLP/rumoureval-2019-training-data/rumoureval-2019-training-data/dev-key.json', 'r') as rlD:
    replyLabelDev = json.load(rlD)


# location of file where all tweets will be stored (both Dev and Train)
with open(r"C:/Users/palak/Desktop/CMSC516_NLP/rumoureval-2019-training-data/twitter-english", 'w') as fout:
    for root, subFolders, files in os.walk(folder):
        tweetText = {}
        tweetTag = {}
        if 'structure.json' in files:
            with open(os.path.join(root, 'structure.json'), 'r') as fin:
                srcTwtTxt = ""
                for line in fin:
                    k = re.findall(r"[\d]+", line) 
                    #re.findall(pattern, string, flags=0) Return all non-overlapping matches of pattern in string, as a list of strings. The string is scanned left-to-right, and matches are returned in the order found.
                    for i in range(0, len(k)):
                        if i == 0:
                            try:
                                with open(os.path.join(root, 'source-tweet/' + k[0] + '.json'), 'r') as st:
                                    parsed = json.load(st)
                                    srcTwtTxt = ' '.join(parsed['text'].splitlines())
                                    tokens = tknzr.tokenize(srcTwtTxt)
                                    maxLengthTweet.append(len(tokens))
                                    srcTwtTxt = ' '.join(tokens)
                                    tweetText[k[0]] = srcTwtTxt
                            except:
                                print("error in getting source tweet file " + k[0])  # printing
                        else:
                            try:
                                with open(os.path.join(root, 'replies/' + k[i] + '.json'), 'r') as st:
                                    parsed = json.load(st)
                                    text = ' '.join(('' + parsed['text']).splitlines())
                                    tokens = tknzr.tokenize(text)
                                    maxLengthTweet.append(len(tokens))
                                    text = ' '.join(tokens)
                                    tweetText[k[i]] = text
                            except:
                                print("error in getting reply tweet file " + k[i])
                                    
                        if k[i] in replyLabelTrain:
                            tweetTag[k[i]] = replyLabelTrain[k[i]]
                        elif k[i] in replyLabelDev:
                            tweetTag[k[i]] = replyLabelDev[k[i]]

        if 'structureFormatted.json' in files:
            with open(os.path.join(root, 'structureFormatted.json'), 'r') as fin:
                stack_twt = []
                level = 0
                for line in fin:
                    k = re.findall(r"[\d]+", line)
                    if '{' in line:
                        if not k:
                            continue
                        else:
                            try:
                                stack_twt.append(k[0])
                                level += 1
                                fout.write(str(level) + '\t' + tweetTag[k[0]])
                                for i in range(0, len(stack_twt)):
                                    fout.write('\t' + tweetText[stack_twt[i]])
                                fout.write('\n')
                            except:
                                print(k[0]+'\n')

                    elif '}' in line:
                        if not stack_twt:
                            continue
                        else:
                            stack_twt.pop()
                            level -= 1
                    elif '[]' in line:
                        try:
                            level += 1
                            fout.write(str(level) + '\t' + tweetTag[k[0]])
                            for i in range(0, len(stack_twt)):
                                fout.write('\t' + tweetText[stack_twt[i]])
                            fout.write('\t' + tweetText[k[0]] + '\n')
                            level -= 1
                        except:
                            print(k[0]+'\n')