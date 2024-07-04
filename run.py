import json
import csv
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import math
from sklearn.preprocessing import LabelEncoder
from pathlib import Path



#Approaches
#i) data: all columns except wardrobe_id, labels =1 if trisha likes
#ii) Make y multi-dimensional for each feature?
#wardrobe_id = qid
#wardrobe_created_at_ms same as wardrobe_id
#iii) Add price range
#LabelEncoder doesn't rank based on numerical values
# Onehotencoder will increase dimensions so no
#Range of 500


#mean
#lambdarank_num_pair_per_sample= 7930
#7940 within 14
#7832 within 15
#7950 within 13
#7925 within 13
#7929 within 12
#7926
#mean performs much better than topk
#qids are only used in trainings


def round_to_nearest_500_smaller(number):
  rounded_number = math.floor(number / 500)
  rounded_number *= 500
  return rounded_number

def generate_dress_recommendations(user_data, dress_data_source):
    enc = LabelEncoder()
    if 'dress_recommendations' not in os.getcwd():
        os.chdir('.\dress_recommendations')

    dataframe = pd.read_csv(dress_data_source)
    f = open(user_data)
    name = user_data.split('.')[0]

    pref = json.load(f)
    dress_pref = pref['last_saved_dresses'] #ids for the dresses
    dress_pref = [int(x.replace('wardrobe-','') ) for x in dress_pref]
    #dataframe['wardrobe_id'] = dataframe['wardrobe_id'].str.replace('wardrobe-', '').astype(np.int64)
    dataframe = dataframe.loc[dataframe['wardrobe_size'] >=0]
    dataframe['wardrobe_price'] = dataframe['wardrobe_price'].apply(round_to_nearest_500_smaller)
    dataframe1 = dataframe.copy()  
    dataframe1['label'] = dataframe1['wardrobe_created_at_ms'].apply(lambda x: 31 if x in dress_pref else 0) # Making 'labels' column
    wardrobe_id = [int(x.replace('wardrobe-','') ) for x in dataframe1['wardrobe_id']]

    ranker = xgb.XGBRanker(tree_method="hist", lambdarank_num_pair_per_sample=7926, objective="rank:ndcg", lambdarank_pair_method="mean")
    for col in dataframe1.columns:
        if col not in ['label','wardrobe_price','wardrobe_id','wardrobe_created_at_ms','wardrobe_photo_signature','wardrobe_size']:
            dataframe1[col] = enc.fit_transform(dataframe1[col])
    
    #dataframe1['qid'] = y
    #dataframe1 = dataframe1.sort_values(by='qid')
    y = dataframe1['label']
    dataframe1 = dataframe1.drop(columns=['label','wardrobe_id','wardrobe_created_at_ms','wardrobe_photo_signature']) #dropping wardrobe_photo_signature and wardrobe_created_at_ms will give us features we care about
    df = pd.DataFrame(0, index  = np.arange(dataframe1.shape[0]),columns=['y']) 
    ranker.fit(dataframe1, y, qid = df)

    #Need qid during training and not inference
    #dropping label for prediction

    scores = ranker.predict(dataframe1)
    dataframe1['scores'] = scores
    dataframe1['wardrobe_id'] = wardrobe_id
    dataframe1.sort_values(by=['scores'], ascending = False, inplace=True)

    with open(name+'.txt',"w") as f:
            li = [str(x) for x in dataframe1['wardrobe_id']][:100]
            f.write('\n'.join(li))
            dataframe1.to_csv(name+'.csv')
    print(dataframe1)
    

generate_dress_recommendations("trisha.json", "wardrobe_items_downloaded_data.csv")
