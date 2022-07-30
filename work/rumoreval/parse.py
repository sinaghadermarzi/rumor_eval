import os
import json

TRAIN_KEY_PATH = '/Users/emiljoswin/study/VCU/ANLP/RumourEval/rumoureval-2019-trial-data/train-key.json'
DEV_KEY_PATH = '/Users/emiljoswin/study/VCU/ANLP/RumourEval/rumoureval-2019-trial-data/dev-key.json'

TOPICS = ['charliehebdo', 
  'ferguson',
  'illary',
  'prince-toronto',
  'sydneysiege',
  'ebola-essien',
  'germanwings-crash',
  'ottawashooting',
  'putinmissing'
]

formatted = {} #where the output will be
sdqc_labels = {}
veracity_lables = {}

FOLDER = '/Users/emiljoswin/study/VCU/ANLP/RumourEval/rumoureval-2019-trial-data/twitter-english/'



def load_labels():
  with open(TRAIN_KEY_PATH) as f:
    data = json.load(f)

  for key in data['subtaska']:
    sdqc_labels[key] = data['subtaska'][key]

  for key in data['subtaskb']:
    veracity_lables[key] = data['subtaskb'][key]
    
  with open(DEV_KEY_PATH) as f:
    data = json.load(f)

  for key in data['subtaska']:
    sdqc_labels[key] = data['subtaska'][key]

  for key in data['subtaskb']:
    veracity_lables[key] = data['subtaskb'][key]

 
def format_topic(t):
  formatted[t] = []
  dir = FOLDER + t

  sub_dirs = os.listdir(dir)

  for sub_dir in sub_dirs:
    
    source_tweet = {}
    reply_tweets = []
    source_path = dir + '/' + sub_dir + '/source-tweet/' + sub_dir + ".json"
    data = {}
    with open(source_path) as f:
      data = json.load(f)

    source_tweet['text'] = data['text']
    source_tweet['id_str'] = data['id_str']
    source_tweet['label'] = veracity_lables[data['id_str']]

    reply_dir =  dir + '/' + sub_dir + '/replies'
    reply_paths = os.listdir(reply_dir)

    for reply_path in reply_paths:
      json_file = reply_dir + '/' + reply_path
      with open(json_file) as f:
        data = json.load(f)

      reply_tweets.append({'text': data['text'], 'id_str': data['id_str'], 'label': sdqc_labels[data['id_str']]})

    formatted[t].append({'source': source_tweet, 'replies': reply_tweets })    

  # print(json.dumps(formatted, indent=4) )

def load_parsed_data_into_files():
  for t in TOPICS:
    format_topic(t)

    f_name = t + '.json'
    dictionary = {t: formatted[t]}

    print("creating file", f_name)
    with open(f_name, 'w') as filename:
      json.dump(dictionary, filename, indent=4)


load_labels()
load_parsed_data_into_files()





