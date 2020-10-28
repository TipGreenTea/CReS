"""
GW Data Pre-processing
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from numpy import array
import json
import seaborn as sns
from statistics import median,median_low,median_high,median_grouped,mean
import statistics 
from datetime import datetime, timezone, timedelta
import json
"""
0. Parameter Settings
"""
FILE_JSON = 'datasets/gw/gowalla_category_structure.json'
FILE_SPOT_SUBSET = 'datasets/gw/gowalla_spots_subset1.csv'
FILE_GENRE_JSON = 'datasets/gw/poi_genre_json.csv'
FILE_GENRE = 'datasets/gw/poi_genre.csv'
FILE_CHECKIN = 'datasets/gw/gowalla_checkins.csv'
MAX_SESSION_ITEM = 50
"""
1. Transform Json into the Categorical List
"""
with open(FILE_JSON) as f:
  data = json.load(f) 
dfObj = pd.DataFrame(columns=['genre', 'subgenre', 'url'])
main_genre = ''
#-----------------1. Community--------------------#
community = data['spot_categories'][0]
main_genre = community['name']
dfObj = dfObj.append({'genre': main_genre, 'subgenre': main_genre, 'url': community['url']}, ignore_index=True)
com = community['spot_categories'] 
for i in range(len(com)):   #7
    for key, value in com[i].items():   #6 keys
        sub_com = com[i]['spot_categories']
        dfObj = dfObj.append({'genre': main_genre, 'subgenre': com[i]['name'], 'url': com[i]['url']}, ignore_index=True)
        for j in range(len(sub_com)):
            dfObj = dfObj.append({'genre': main_genre, 'subgenre': sub_com[j]['name'], 'url': sub_com[j]['url']}, ignore_index=True)
#-----------------2. Entertainment--------------------#
entertainment = data['spot_categories'][1]
main_genre = entertainment['name'] 
dfObj = dfObj.append({'genre': main_genre, 'subgenre': main_genre, 'url': entertainment['url']}, ignore_index=True)
en = entertainment['spot_categories'] #---> get the list
for i in range(len(en)):   #7
    for key, value in en[i].items():   #6 keys
        sub_en = en[i]['spot_categories']
        dfObj = dfObj.append({'genre': main_genre, 'subgenre': en[i]['name'], 'url': en[i]['url']}, ignore_index=True)
        for j in range(len(sub_en)):
            dfObj = dfObj.append({'genre': main_genre, 'subgenre': sub_en[j]['name'], 'url': sub_en[j]['url']}, ignore_index=True)
#-----------------3. Food--------------------#
food = data['spot_categories'][2]
main_genre = food['name']
dfObj = dfObj.append({'genre': main_genre, 'subgenre': main_genre, 'url': food['url']}, ignore_index=True)
foo = food['spot_categories'] #---> get the list
for i in range(len(foo)):   #7
    for key, value in foo[i].items():   #6 keys
        sub_foo = foo[i]['spot_categories']
        dfObj = dfObj.append({'genre': main_genre, 'subgenre': foo[i]['name'], 'url': foo[i]['url']}, ignore_index=True)
        for j in range(len(sub_foo)):
            dfObj = dfObj.append({'genre': main_genre, 'subgenre': sub_foo[j]['name'], 'url': sub_foo[j]['url']}, ignore_index=True)
#-----------------4. Nightlife--------------------#
nightlife = data['spot_categories'][3]
main_genre = nightlife['name']
dfObj = dfObj.append({'genre': main_genre, 'subgenre': main_genre, 'url': nightlife['url']}, ignore_index=True)
night = nightlife['spot_categories'] #---> get the list 
for i in range(len(night)):   #7
    for key, value in night[i].items():   #6 keys
        sub_night = night[i]['spot_categories']
        dfObj = dfObj.append({'genre': main_genre, 'subgenre': night[i]['name'], 'url': night[i]['url']}, ignore_index=True)
        for j in range(len(sub_night)):
            dfObj = dfObj.append({'genre': main_genre, 'subgenre': sub_night[j]['name'], 'url': sub_night[j]['url']}, ignore_index=True)
#-----------------5. Outdoors--------------------#
outdoors = data['spot_categories'][4]
main_genre = outdoors['name']
dfObj = dfObj.append({'genre': main_genre, 'subgenre': main_genre, 'url': outdoors['url']}, ignore_index=True)
out = outdoors['spot_categories'] #---> get the list
for i in range(len(out)):   #7
    for key, value in out[i].items():   #6 keys
        sub_out = out[i]['spot_categories']
        dfObj = dfObj.append({'genre': main_genre, 'subgenre': out[i]['name'], 'url': out[i]['url']}, ignore_index=True)
        for j in range(len(sub_out)):
            dfObj = dfObj.append({'genre': main_genre, 'subgenre': sub_out[j]['name'], 'url': sub_out[j]['url']}, ignore_index=True)
#-----------------6. Shopping--------------------#
shoppings = data['spot_categories'][5]
main_genre = shoppings['name']
dfObj = dfObj.append({'genre': main_genre, 'subgenre': main_genre, 'url': shoppings['url']}, ignore_index=True)
shop = shoppings['spot_categories'] #---> get the list
for i in range(len(shop)):   #7
    for key, value in shop[i].items():   #6 keys
        sub_shop = shop[i]['spot_categories']
        dfObj = dfObj.append({'genre': main_genre, 'subgenre': shop[i]['name'], 'url': shop[i]['url']}, ignore_index=True)
        for j in range(len(sub_shop)):
            dfObj = dfObj.append({'genre': main_genre, 'subgenre': sub_shop[j]['name'], 'url': sub_shop[j]['url']}, ignore_index=True)
#-----------------7. Travel--------------------#
travels = data['spot_categories'][6]
main_genre = travels['name']
dfObj = dfObj.append({'genre': main_genre, 'subgenre': main_genre, 'url': travels['url']}, ignore_index=True)
trav = travels['spot_categories'] #---> get the list
for i in range(len(trav)):   #7
    for key, value in trav[i].items():   #6 keys
        sub_trav = trav[i]['spot_categories']
        dfObj = dfObj.append({'genre': main_genre, 'subgenre': trav[i]['name'], 'url': trav[i]['url']}, ignore_index=True)
        for j in range(len(sub_trav)):
            dfObj = dfObj.append({'genre': main_genre, 'subgenre': sub_trav[j]['name'], 'url': sub_trav[j]['url']}, ignore_index=True)
dfObj = dfObj.drop_duplicates() 
dfObj.to_csv(FILE_GENRE_JSON, sep='\t', index=False)
"""
2. merge 2 dataframes and produce poi_genre.csv
"""
df_subset = pd.read_csv(FILE_SPOT_SUBSET,sep=',' ,skiprows=1, header=None, index_col=False)
df_genre = pd.read_csv(FILE_GENRE_JSON,sep='\t' ,skiprows=1, header=None, index_col=False)
df_subset.columns = ['poiId','created_at','lng','lat','photos_count','checkins_count','users_count','radius_meters','highlights_count','items_count','max_items_count','cat']
df_genre.columns = ['genre', 'subgenre', 'url']
df_subset['url'] = '0'
for i in df_subset.index:
    updated_url = df_subset.at[i, "cat"].split(',')[0].split(':')[1].replace('\'','').replace(' ','')
    df_subset.at[i, "url"] = updated_url

del df_subset['created_at'], df_subset['lng'], df_subset['lat'], df_subset['photos_count'], df_subset['checkins_count'], df_subset['users_count'], df_subset['radius_meters'], df_subset['highlights_count'], df_subset['items_count'], df_subset['max_items_count'], df_subset['cat']

df_merge = pd.merge(df_subset, df_genre, on='url')
df_merge.to_csv(FILE_GENRE, sep='\t', index=False)

"""
3. Compute gowalla_checkins.csv and Sessionize
"""

df_checkin = pd.read_csv(FILE_CHECKIN,sep=',' ,skiprows=1, header=None, index_col=False)
df_checkin.columns = ['UserId', 'ItemId', 'Datetime'] #Time
print("Item: {} User:{} ".format(df_checkin.ItemId.nunique(), df_checkin.UserId.nunique()))

df_checkin['Time'] = df_checkin['Datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
df_checkin['Time'] = df_checkin['Time'].apply(lambda x: datetime.timestamp(x))
df_checkin['TimeTmp'] = df_checkin['Time'].apply(lambda x: datetime.fromtimestamp(x))

df_checkin.sort_values( ['UserId','TimeTmp'], ascending=True, inplace=True )
df_checkin['TimeShift'] = df_checkin['TimeTmp'].shift(1)
df_checkin['TimeDiff'] = (df_checkin['TimeTmp'] - df_checkin['TimeShift']).dt.total_seconds().abs()
df_checkin['Session_Threshold'] = 86400     #1day
df_checkin['SessionIdTmp'] = (df_checkin['TimeDiff'] > df_checkin['Session_Threshold']).astype( int )
df_checkin['SessionId'] = df_checkin['SessionIdTmp'].cumsum( skipna=False )
del df_checkin['Session_Threshold'], df_checkin['SessionIdTmp'], df_checkin['TimeShift'], df_checkin['TimeDiff'], df_checkin['Datetime']
df_checkin.to_csv('datasets/gw_poi_dataset_full.txt', sep='\t', index=False)

"""
4. Re-Sessionize based on MAX_SESSION_ITEM (poi_dataset_full.txt) 
"""
df = pd.read_csv('datasets/gw_poi_dataset_full.txt',sep='\t',skiprows=1, header=None, index_col=False)
df.columns = ['UserId', 'ItemId', 'Time','Timetmp','SessionId']
df['SessIdShift'] = df['SessionId'].shift(1)
df['item_num'] = df.groupby(['UserId','SessionId']).cumcount() +1
df['n_inx']= df['item_num']/MAX_SESSION_ITEM
df['n_inx']= df['n_inx'].apply(np.int64)
df['n_inx_shf'] = df['n_inx'].shift(1)

df['IdTmp'] = (df['n_inx'] != df['n_inx_shf']).astype( int )
df['IdTmp2'] = (df['SessionId'] != df['SessIdShift']).astype(int)
df['IdTmp3'] = df['IdTmp'] + df['IdTmp2']
df.loc[df.IdTmp3 == 2, 'IdTmp3'] = 1
df['Final_SessId'] = df['IdTmp3'].cumsum( skipna=False )

del df['SessIdShift'], df['item_num'], df['n_inx'], df['n_inx_shf'], df['IdTmp'], df['IdTmp2'], df['IdTmp3']

data_start = datetime.fromtimestamp( df.Time.min(), timezone.utc )
data_end = datetime.fromtimestamp( df.Time.max(), timezone.utc )
    
df.to_csv('datasets/gw_poi_dataset_full.txt', sep='\t', index=False)

"""
5. Filter Dataset (poi_dataset_full.txt) 
"""

MAX_SESSION_ITEM = 50
MIN_ITEM_SUPPORT = 20 
FILE_OUTPUT_ITEM_SUPPORT = 'item_support_'+str(5)+'/' 
MIN_SESSION_LENGTH = 15 
MIN_USER_LENGTH = 20 #
FILE_NAME = 'datasets/gw_poi_dataset_full.txt'
FLAG = 1
   
df = pd.read_csv(FILE_NAME,sep='\t',skiprows=1, header=None, index_col=False)
df.columns = ['UserId', 'ItemId', 'Time','Timetmp','x','SessionId']
del df['x']

while(FLAG):
    session_lengths = df.groupby('SessionId').size()
    df = df[np.in1d(df.SessionId, session_lengths[ session_lengths>= MIN_SESSION_LENGTH ].index)]    
    #filter item support
    item_supports = df.groupby('ItemId').size()
    df = df[np.in1d(df.ItemId, item_supports[ item_supports>= MIN_ITEM_SUPPORT ].index)]

    #filter session length
    session_lengths = df.groupby('SessionId').size()
    df = df[np.in1d(df.SessionId, session_lengths[ session_lengths>= MIN_SESSION_LENGTH ].index)]

    #filter user length
    user_lengths = df.groupby('UserId')['SessionId'].nunique()
    df = df[np.in1d(df.UserId, user_lengths[ user_lengths>= MIN_USER_LENGTH ].index)]

    #calculate the updated df
    item_supports = df.groupby('ItemId').size()
    session_lengths = df.groupby('SessionId').size()
    user_lengths = df.groupby('UserId')['SessionId'].nunique()
    if((len(user_lengths[ user_lengths< MIN_USER_LENGTH ].index) == 0) & 
       (len(session_lengths[ session_lengths< MIN_SESSION_LENGTH ].index) == 0) & 
       (len(item_supports[ item_supports< MIN_ITEM_SUPPORT ].index) == 0)):
        FLAG = 0
        
#output
data_start = datetime.fromtimestamp( df.Time.min(), timezone.utc )
data_end = datetime.fromtimestamp( df.Time.max(), timezone.utc )
    
df.to_csv('datasets/gw_poi_dataset_filter.txt', sep='\t', index=False)   
with open('datasets/gw_poi_dataset_filter_stat.txt', "w") as text_file:
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tUsers: {} \n\tItems: {} \n\tAvg. Users Session Length: {} \n\tAvg. Sessions Item Length: {} \n\tSpan: {} / {}\n\n'.
      format( len(df), df.SessionId.nunique(), df.UserId.nunique(), df.ItemId.nunique(), 
             min(user_lengths.values),mean(user_lengths.values),max(user_lengths.values), 
             median_low(user_lengths.values),median_grouped(user_lengths.values),median_high(user_lengths.values),
             min(session_lengths.values),mean(session_lengths.values), max(session_lengths.values),
            median_low(session_lengths.values),median_grouped(session_lengths.values), median_high(session_lengths.values),
             data_start.date().isoformat(), data_end.date().isoformat() ), file=text_file)

"""
6. Prepare file for Item Embedding Training (item_seq)
Consider all items in dataset after filtering
"""

MAX_SESSION_ITEM = 50
MIN_ITEM_SUPPORT = 5
FILE_INPUT = 'datasets/gw_poi_dataset_filter.txt'
FILE_OUTPUT = 'datasets/gw_poi_dataset_item_seq.txt'
df = pd.read_csv(FILE_INPUT,sep='\t',skiprows=1, header=None, index_col=False)
df.columns = ['UserId', 'ItemId', 'Time','Timetmp','SessionId']
df_sort = df.sort_values(['UserId', 'Time'], ascending=[True, True])
row = 0
output = ""
current_user = ''
for sample in df_sort.iterrows(): 
    if row == 1:
        current_user = sample[1]['UserId']
        output += str(sample[1]['ItemId']) + " " 
    elif row >1:
        if current_user != sample[1]['UserId']: 
            current_user = sample[1]['UserId']
            output += "\n"
            output += str(sample[1]['ItemId']) + " "
        else:
            output += str(sample[1]['ItemId']) + " " 
    row +=1

file = open(FILE_OUTPUT,"w") 
file.write(output)  
file.close() 

"""
7. Seperate dataset to K
"""

MAX_SESSION_ITEM = 50
MIN_ITEM_SUPPORT = 5
FILE_INPUT = 'datasets/gw_poi_dataset_filter.txt'
K = [10]
df = pd.read_csv(FILE_INPUT,sep='\t',skiprows=1, header=None, index_col=False)
df.columns = ['UserId','ItemId','Time','Timetmp','SessionId']
df.sort_values( ['UserId','SessionId','Time'], ascending=True, inplace=True )
user_lengths = df.groupby('UserId')['SessionId'].nunique()  #how many sessions each user has

unique_user = df.UserId.unique()
for i in range(len(unique_user)):
    df.loc[df['UserId'] == unique_user[i], 'K_10'] = np.round((user_lengths.values[i] >= 12).astype(int))

for i in range(len(K)):
    FILE_OUTPUT = 'datasets/gw_data_df.txt'
    df = df[df['K_'+str(K[i])] == 1]
    df.sort_values( ['UserId','SessionId','Time'], ascending=True, inplace=True )
    df.to_csv(FILE_OUTPUT, sep='\t', index=False)
    data_start = datetime.fromtimestamp( df.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( df.Time.max(), timezone.utc )
    with open('datasets/gw_stat.txt', "w") as text_file:
        print('data set\n\tEvents: {}\n\tSessions: {}\n\tUsers: {} \n\tItems: {}  \n\tSpan: {} / {}\n\n'.
              format( len(df), df.SessionId.nunique(), df.UserId.nunique(), df.ItemId.nunique(), 
             data_start.date().isoformat(), data_end.date().isoformat() ), file=text_file)

"""
8. Seperate K dataset to instances (train/test)
"""

MAX_SESSION_ITEM = 50
MIN_ITEM_SUPPORT = 5
TRAIN_PERCENTAGE = .9
K = [10]
train_df = pd.DataFrame(columns = ['UserId', 'ItemId','Time','SessionId'])
test_df = pd.DataFrame(columns = ['UserId', 'ItemId','Time','SessionId'])
for k in K:
    FILE_INPUT = 'datasets/gw_data_df.txt'
    FILE_STAT = 'datasets/gw_stat_traintest_instances.txt'
    file_stat= open(FILE_STAT,"w+")
    df = pd.read_csv(FILE_INPUT,sep='\t',skiprows=1, header=None, index_col=False)
    df.columns = ['UserId','ItemId','Time','Timetmp','SessionId','1', '2']
    del df['1'], df['2']
    df = df.sort_values( ['UserId','SessionId','Time'], ascending=[True, True,True])
    
    user_lengths = df.groupby('UserId')['SessionId'].nunique()  #how many sessions each user has
    
    unique_user = df.UserId.unique()
    
    training_long_instance = []
    training_short_instance = []
    testing_long_instance = []
    testing_short_instance = []
    training_user = []
    testing_user = []
    for u in range(len(unique_user)):
        current_df = df[df['UserId'] == unique_user[u]]
        total_instances = current_df.SessionId.nunique()    #X
        current_sessionId_list = current_df.SessionId.unique()
        total_instances = total_instances-k #X-K
        total_train_instances = int(total_instances * TRAIN_PERCENTAGE)
        total_test_instances = total_instances - total_train_instances

        train_index = 0 
    
        for i in range(len(current_sessionId_list)):    #session index
        
            if(train_index < total_train_instances):
                K_input_session = []
                for j in range(k):  #K
                    current_session = current_df[current_df['SessionId'] == current_sessionId_list[i+j]]
                    #items in this session
                    K_input_session.append(current_session['ItemId'].values.tolist())
                    
                    
                training_long_instance.append(K_input_session)
            
                short_session = current_df[current_df['SessionId'] == current_sessionId_list[i+k]]
                
                train_df = train_df.append(short_session, ignore_index=True)#
                
                training_short_instance.append(short_session['ItemId'].values.tolist())
                training_user.append(current_df['UserId'].unique()[0])
            elif(train_index >= total_train_instances and train_index <total_instances):
                #print("train_index: ", train_index)
                K_input_session = []
                for j in range(k):  #K
                    current_session = current_df[current_df['SessionId'] == current_sessionId_list[i+j]]
                    #items in this session
                    K_input_session.append(current_session['ItemId'].values.tolist())
                    
                testing_long_instance.append(K_input_session)
                short_session = current_df[current_df['SessionId'] == current_sessionId_list[i+k]]
                
                test_df = test_df.append(short_session, ignore_index=True)#
                
                testing_short_instance.append(short_session['ItemId'].values.tolist())
                testing_user.append(current_df['UserId'].unique()[0])
            train_index +=1
        file_stat.write('UserId: {} \tTotalSessInstances: {}  \tTrain: {} \tTest: {}\n'.
              format(current_df['UserId'].unique()[0],total_instances,total_train_instances,total_test_instances))
        file_stat.flush()
       
    #write to file
    FILE_TRAINING_USER = 'datasets/gw_training_user.txt'
    FILE_TRAINING_LONG = 'datasets/gw_training_long_instance.txt'
    FILE_TRAINING_SHORT = 'datasets/gw_training_short_instance.txt'
    FILE_TRAINING_DF = 'datasets/gw_train.txt'
    FILE_TESTING_USER = 'datasets/gw_testing_user.txt'
    FILE_TESTING_LONG = 'datasets/gw_testing_long_instance.txt'
    FILE_TESTING_SHORT = 'datasets/gw_testing_short_instance.txt'
    FILE_TESTING_DF = 'datasets/gw_test.txt'
    file_training_user = open(FILE_TRAINING_USER,"w+")
    file_training_long = open(FILE_TRAINING_LONG,"w+")
    file_training_short = open(FILE_TRAINING_SHORT,"w+")
    file_testing_user = open(FILE_TESTING_USER,"w+")
    file_testing_long = open(FILE_TESTING_LONG,"w+")
    file_testing_short = open(FILE_TESTING_SHORT,"w+")
    
    file_training_user.write('\n'.join(str(line) for line in training_user))
    file_training_long.write('\n'.join(str(line) for line in training_long_instance))
    file_training_short.write('\n'.join(str(line) for line in training_short_instance))
    file_training_user.flush(); file_training_long.flush(); file_training_short.flush();
    train_df.to_csv(FILE_TRAINING_DF, sep='\t')
    file_testing_user.write('\n'.join(str(line) for line in testing_user))
    file_testing_long.write('\n'.join(str(line) for line in testing_long_instance))
    file_testing_short.write('\n'.join(str(line) for line in testing_short_instance))
    file_testing_user.flush(); file_testing_long.flush(); file_testing_short.flush()
    test_df.to_csv(FILE_TESTING_DF, sep='\t')
    
    file_stat.close(); file_training_user.close(); file_training_long.close()
    file_training_short.close(); file_testing_user.close(); file_testing_long.close(); 
    file_testing_short.close(); #file_training_df.close(); file_testing_df.close()

"""
9. Create All Embedding Variables
"""

import gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#import json
from numpy import savetxt, loadtxt
import numpy as np
import pandas as pd
MIN_ITEM_SUPPORT = [5]   
MAX_SESSION_ITEM = [50]


for m in MIN_ITEM_SUPPORT:
    for n in MAX_SESSION_ITEM:
        
        FILE_ITEM_SEQ = 'datasets/gw_poi_dataset_item_seq.txt'
        FILE_USER_UNIQUE = 'datasets/gw_poi_dataset_filter.txt'
        EMBEDDING_ITEM_OUTPUT = 'datasets/gw_item_embedding'
        EMBEDDING_USER_OUTPUT = 'datasets/gw_user_embedding.csv'
        EMBEDDING_CONTEXT_OUTPUT = 'datasets/gw_context_embedding.csv'
        EMBEDDING_CONTEXTPERCENT_OUTPUT = 'datasets/gw_contextpercent_embedding.csv'
        EMBEDDING_USER_INDEX_DICT = 'datasets/gw_user_index_dict.txt'
        EMBEDDING_INDEX_USER_DICT = 'datasets/gw_index_user_dict.txt'
        EMBEDDING_CONTEXT_INDEX_DICT = 'datasets/gw_context_index_dict.txt'
        EMBEDDING_INDEX_CONTEXT_DICT = 'datasets/gw_index_context_dict.txt'
        EMBEDDING_CONTEXTPERCENT_INDEX_DICT = 'datasets/gw_contextpercent_index_dict.txt'
        EMBEDDING_INDEX_CONTEXTPERCENT_DICT = 'datasets/gw_index_contextpercent_dict.txt'
        
        EMBEDDING_INPUT_DIM = 32 
        EMBEDDING_EPOCH =   20    
        EMBEDDING_WINDOW =  10   
        EMBEDDING_MIN_COUNT = 1  
        EMBEDDING_WORKER =  4    
        
        CONTEXTS = ['Community','Entertainment', 'Food', 'Nightlife', 'Outdoors', 'Shopping', 'Travel']    #context 1-7    
        PERCENTS = ['1','2','3','4','5']   
        FILE_STAT = 'datasets/gw_poi_dataset_filter_stat.txt'
        stat = open(FILE_STAT, "r"); stat_lines = (stat.readlines())
        total_user = int(stat_lines[3].replace(" ","").replace("\n","").split(":")[1]) #TOTAL_USER = 1854
       
        total_context = len(CONTEXTS)
        total_percent = len(PERCENTS)
        
        #Item Embedding Initialization
        input_file = FILE_ITEM_SEQ  #varaibles
        sentences = []
        index = 0
        file = open(input_file, "r") 
        for line in file: 
            line = line.split(" ")
            line = line[:-1] 
            sentences.append(line)
            index +=1
        file.close()
        
       
        model = gensim.models.Word2Vec(sentences, size=EMBEDDING_INPUT_DIM, window=EMBEDDING_WINDOW, min_count=EMBEDDING_MIN_COUNT, workers=EMBEDDING_WORKER)
        model.train(sentences,total_examples=len(sentences),epochs=EMBEDDING_EPOCH)
        ##------------------Save--------------------##
        model.save(EMBEDDING_ITEM_OUTPUT)
        
        pretrained_weights = model.wv.vectors
        vocab_size, emdedding_test_size = pretrained_weights.shape
        
        pretrained_weights_padded =np.vstack([np.zeros((EMBEDDING_INPUT_DIM)), pretrained_weights])
        vocab_size_padded = vocab_size +1
        pretrained_weights_padded = np.asarray(pretrained_weights_padded, dtype=np.float32)
        #****have to be used for every models****#
        def word2idx(word):
            return (model.wv.vocab[word].index) +1    #don't have index =0 
        def idx2word(idx):
            return model.wv.index2word[idx-1]
        
        #User & Context & Context Percentage Embedding Initialization
        user_embedding = np.random.normal(0,.1,(total_user,EMBEDDING_INPUT_DIM))
        savetxt(EMBEDDING_USER_OUTPUT, user_embedding, delimiter=',')    # save array
        context_embedding = np.random.normal(0,.1,(total_context,EMBEDDING_INPUT_DIM))
        savetxt(EMBEDDING_CONTEXT_OUTPUT, context_embedding, delimiter=',')      # save array
        context_percent_embedding = np.random.normal(0,.1,(total_percent,EMBEDDING_INPUT_DIM))
        savetxt(EMBEDDING_CONTEXTPERCENT_OUTPUT, context_percent_embedding, delimiter=',')      # save array
        
        """
        User & Context & Context Percentage Save Dictionary
        """
        """
        Dictionary Save & Store
        """
        #-------------Save Dictioanry (Word <--> Index)---------------#
        def save_dict_to_file(file,dic):
            f = open(file,'w')
            f.write(str(dic))
            f.close()
        
        def load_dict_from_file(file):
            f = open(file,'r')
            data=f.read()
            f.close()
            return eval(data)
        
        #-------------Save User Dictionary---------------#
        df = pd.read_csv(FILE_USER_UNIQUE, sep='\t',skiprows=1, header=None, index_col=False) 
        df.columns = ['UserId','ItemId','Time','Timetmp','SessionId']
        
        unique_users = df.UserId.unique()
        user_index = {user: idx for idx, user in enumerate(unique_users)}
        index_user = {idx: user for user, idx in user_index.items()}
        save_dict_to_file(EMBEDDING_USER_INDEX_DICT, user_index)
        save_dict_to_file(EMBEDDING_INDEX_USER_DICT, index_user)
        
        #-------------Save Context Dictionary---------------#
        context_index = {context: idx for idx, context in enumerate(CONTEXTS)}
        index_context = {idx: context for context, idx in context_index.items()}
        save_dict_to_file(EMBEDDING_CONTEXT_INDEX_DICT, context_index)
        save_dict_to_file(EMBEDDING_INDEX_CONTEXT_DICT, index_context)
        
        #-------------Save Context Dictionary---------------#
        contextpercent_index = {percent: idx for idx, percent in enumerate(PERCENTS)}
        index_contextpercent = {idx: percent for percent, idx in contextpercent_index.items()}
        save_dict_to_file(EMBEDDING_CONTEXTPERCENT_INDEX_DICT, contextpercent_index)
        save_dict_to_file(EMBEDDING_INDEX_CONTEXTPERCENT_DICT, index_contextpercent)



"""
10. Transform instances into model inputs
Split data (ml_dataset_filter.txt) for Baselines
Based on Users' sessions Order & Time
https://stackoverflow.com/questions/46784265/reset-cumulative-sum-base-on-condition-pandas
"""

print("Transform instances into model inputs")
import gensim 
from numpy import savetxt, loadtxt
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import pickle
CONTEXTS = ['Community','Entertainment', 'Food', 'Nightlife', 'Outdoors', 'Shopping', 'Travel']    #context 1-7        
PERCENTS = ['1','2','3','4','5']    #fixed at the beginning
EMBEDDING_INPUT_DIM = 32
MIN_ITEM_SUPPORT = [5]
MAX_SESSION_ITEM = [50]
K= [10] 

for m in MIN_ITEM_SUPPORT:
    for n in MAX_SESSION_ITEM:
        EMBEDDING_ITEM_OUTPUT = 'datasets/gw_item_embedding'
        EMBEDDING_USER_OUTPUT = 'datasets/gw_user_embedding.csv'
        EMBEDDING_CONTEXT_OUTPUT = 'datasets/gw_context_embedding.csv'
        EMBEDDING_CONTEXTPERCENT_OUTPUT = 'datasets/gw_contextpercent_embedding.csv'
        USER_INDEX = 'datasets/gw_user_index_dict.txt'
        INDEX_USER = 'datasets/gw_index_user_dict.txt'
        CONTEXT_INDEX = 'datasets/gw_context_index_dict.txt'
        INDEX_CONTEXT = 'datasets/gw_index_context_dict.txt'
        CONTEXTPERCENT_INDEX = 'datasets/gw_contextpercent_index_dict.txt'
        INDEX_CONTEXTPERCENT = 'datasets/gw_index_contextpercent_dict.txt'
        #----------pretrained_weights_padded for deep learning model----------#
        #already pad the 0 embedding from data-preprocessing
        embedding_model = gensim.models.Word2Vec.load(EMBEDDING_ITEM_OUTPUT)
        pretrained_weights = embedding_model.wv.vectors
        vocab_size, emdedding_test_size = pretrained_weights.shape
        pretrained_weights_padded =np.vstack([np.zeros((EMBEDDING_INPUT_DIM)), pretrained_weights])
        vocab_size_padded = vocab_size +1
        pretrained_weights_padded = np.asarray(pretrained_weights_padded, dtype=np.float32)
        #****have to be used for every models****#
        def word2idx(word):
            return (embedding_model.wv.vocab[word].index) +1    #don't have index =0 
        def idx2word(idx):
            return embedding_model.wv.index2word[idx-1]
        
        """
        Dictionary Save & Store
        """
        #-------------Save Dictioanry (Word <--> Index)---------------#
        def save_dict_to_file(file,dic):
            f = open(file,'w')
            f.write(str(dic))
            f.close()
            
        def load_dict_from_file(file):
            f = open(file,'r')
            data=f.read()
            f.close()
            return eval(data)
        
        user_embedding = loadtxt(EMBEDDING_USER_OUTPUT, delimiter=',') 
        context_embedding = loadtxt(EMBEDDING_CONTEXT_OUTPUT, delimiter=',') 
        context_percent_embedding = loadtxt(EMBEDDING_CONTEXTPERCENT_OUTPUT, delimiter=',')   
        context_index = load_dict_from_file(CONTEXT_INDEX)
        index_context = load_dict_from_file(INDEX_CONTEXT)
        contextpercent_index = load_dict_from_file(CONTEXTPERCENT_INDEX)
        index_contextpercent = load_dict_from_file(INDEX_CONTEXTPERCENT)
        user_index = load_dict_from_file(USER_INDEX)
        index_user = load_dict_from_file(INDEX_USER)
        
        for k_his in K:
           
            #-----------------------------Methods-------------------------------# 
            
            def calculate_class(value): #get 7 values, return list of context_p
                if value >= 0.81 and value <= 1.00:     return '5' 
                elif value >= 0.61 and value <= 0.80:   return '4' 
                elif value >= 0.41 and value <= 0.60:   return '3' 
                elif value >= 0.21 and value <= 0.40:   return '2' 
                else: return '1' 
                
            #reading file
            FILE_GENRE = 'datasets/gw/poi_genre.csv'
            FILE_TESTING_SHORT = 'datasets/gw_testing_short_instance.txt'
            FILE_TESTING_LONG = 'datasets/gw_testing_long_instance.txt'
            FILE_TESTING_USER = 'datasets/gw_testing_user.txt'
            FILE_TRAINING_SHORT = 'datasets/gw_training_short_instance.txt'
            FILE_TRAINING_LONG = 'datasets/gw_training_long_instance.txt'
            FILE_TRAINING_USER = 'datasets/gw_training_user.txt'
            
            #-----------------------------Variable-------------------------------# 
            test_short_user = []; test_short_output = []; test_short_lastitem = []; test_short_seqlastsession = []
            test_short_context = []; test_short_context_p = []
            test_short_cat1 = []; test_short_cat2 = []; test_short_cat3 = []; test_short_cat4 = []
            test_short_cat5 = []; test_short_cat6 = []; test_short_cat7 = []
            #-----------------------------Variable-------------------------------# 
            test_long_user = [];  test_long_lastitem = []; test_long_context = []; test_long_context_p = []
            test_long_cat1 = []; test_long_cat2 = []; test_long_cat3 = []; test_long_cat4 = []
            test_long_cat5 = []; test_long_cat6 = []; test_long_cat7 = []
            #-----------------------------Variable-------------------------------# 
            train_short_user = []; train_short_output = []; train_short_lastitem = []; train_short_seqlastsession = []
            train_short_context = []; train_short_context_p = []
            train_short_cat1 = []; train_short_cat2 = []; train_short_cat3 = []; train_short_cat4 = []
            train_short_cat5 = []; train_short_cat6 = []; train_short_cat7 = []
            #-----------------------------Variable-------------------------------# 
            train_long_user = [];  train_long_lastitem = []; train_long_context = []; train_long_context_p = []
            train_long_cat1 = []; train_long_cat2 = []; train_long_cat3 = []; train_long_cat4 = []
            train_long_cat5 = []; train_long_cat6 = []; train_long_cat7 = []
            #----------for both short & long term --> train & test in all sessions----------#
            #find max for each cat and pad on the left at the end 
            max_item_cat1= 0; max_item_cat2= 0; max_item_cat3= 0; max_item_cat4= 0
            max_item_cat5= 0; max_item_cat6= 0; max_item_cat7= 0
            #add genre column & transform it to multi-column
            df_genre = pd.read_csv(FILE_GENRE,sep='\t',index_col=False)
            df_genre.columns = ['ItemId','url','genre','subgenre']
            del df_genre['url'], df_genre['subgenre']
           
            df_genre = pd.get_dummies(df_genre, columns=['genre']).groupby(['ItemId'], as_index=False).sum()
           
            
            #---------------------------TESTING_SHORT-----------------------#
            with open(FILE_TESTING_SHORT) as test_short, open(FILE_TESTING_USER) as test_user:
                test_short_lines = test_short.readlines()
                user_lines = test_user.readlines()
                for i in range(len(user_lines)):
                    test_user = user_lines[i].split('\n')
                    
                    test_short_line = test_short_lines[i].replace('\n','').replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')
                    itemId_list = []
                    for l in range(len(test_short_line)):
                        itemId_list.append(test_short_line[l])

                    user = []
                    user.append(user_index.get(int(test_user[0])))
                    
                    output = word2idx(itemId_list[-1:][0])
                    last_item = word2idx(itemId_list[-2:-1][0])
                    current_session = itemId_list[:-1]
                    
                    seqlastsession = [word2idx(str(x)) for x in current_session]
                    
                    test_short_genre = df_genre.loc[df_genre['ItemId'].isin(current_session)]
                    
                    total_items = len(current_session) #if short we exclude the last item
                    
                    total_community = test_short_genre['genre_Community'].sum()
                    if total_community>max_item_cat1: max_item_cat1= total_community
                    tmp_community_items = list(test_short_genre[test_short_genre['genre_Community'] ==1]['ItemId']) #not consider order of item
                    community_items = [word2idx(str(x)) for x in tmp_community_items]
                    if total_community == 0: community_items.append(0)
                    
                    total_entertainment = test_short_genre['genre_Entertainment'].sum()
                    if total_entertainment>max_item_cat2: max_item_cat2= total_entertainment
                    tmp_entertainment_items = list(test_short_genre[test_short_genre['genre_Entertainment'] ==1]['ItemId']) 
                    entertainment_items = [word2idx(str(x)) for x in tmp_entertainment_items]
                    if total_entertainment == 0: entertainment_items.append(0)
                    
                    total_food = test_short_genre['genre_Food'].sum()
                    if total_food>max_item_cat3: max_item_cat3= total_food
                    tmp_food_items = list(test_short_genre[test_short_genre['genre_Food'] ==1]['ItemId']) 
                    food_items = [word2idx(str(x)) for x in tmp_food_items]
                    if total_food == 0: food_items.append(0)
                    
                    total_nightlife = test_short_genre['genre_Nightlife'].sum()
                    if total_nightlife>max_item_cat4: max_item_cat4= total_nightlife
                    tmp_nightlife_items = list(test_short_genre[test_short_genre['genre_Nightlife'] ==1]['ItemId']) 
                    nightlife_items = [word2idx(str(x)) for x in tmp_nightlife_items]
                    if total_nightlife == 0: nightlife_items.append(0)
                    
                    total_outdoors = test_short_genre['genre_Outdoors'].sum()
                    if total_outdoors>max_item_cat5: max_item_cat5= total_outdoors
                    tmp_outdoors_items = list(test_short_genre[test_short_genre['genre_Outdoors'] ==1]['ItemId']) 
                    outdoors_items = [word2idx(str(x)) for x in tmp_outdoors_items]
                    if total_outdoors == 0: outdoors_items.append(0)
                    
                    total_shopping = test_short_genre['genre_Shopping'].sum()
                    if total_shopping>max_item_cat6: max_item_cat6= total_shopping
                    tmp_shopping_items = list(test_short_genre[test_short_genre['genre_Shopping'] ==1]['ItemId']) 
                    shopping_items = [word2idx(str(x)) for x in tmp_shopping_items]
                    if total_shopping == 0: shopping_items.append(0)
                    
                    total_travel = test_short_genre['genre_Travel'].sum()
                    if total_travel>max_item_cat7: max_item_cat7= total_travel
                    tmp_travel_items = list(test_short_genre[test_short_genre['genre_Travel'] ==1]['ItemId']) 
                    travel_items = [word2idx(str(x)) for x in tmp_travel_items]
                    if total_travel == 0: travel_items.append(0)
                    
                    tmp_context_p = [calculate_class(total_community/total_items), calculate_class(total_entertainment/total_items), calculate_class(total_food/total_items),
                                 calculate_class(total_nightlife/total_items), calculate_class(total_outdoors/total_items), calculate_class(total_shopping/total_items), calculate_class(total_travel/total_items)]
                    #transform to index
                    context_p = [contextpercent_index[x] for x in tmp_context_p]
                    
                    context = [context_index[x] for x in CONTEXTS]
                    
                    test_short_user.append(user)
                    test_short_output.append(output)
                    test_short_lastitem.append(last_item)
                    test_short_seqlastsession.append(seqlastsession)
                    test_short_context.append(context)
                    test_short_context_p.append(context_p)
                    test_short_cat1.append(community_items)
                    test_short_cat2.append(entertainment_items)
                    test_short_cat3.append(food_items)
                    test_short_cat4.append(nightlife_items)
                    test_short_cat5.append(outdoors_items)
                    test_short_cat6.append(shopping_items)
                    test_short_cat7.append(travel_items)
            

            
            #-----------------------TESTING_LONG-----------------------#
            with open(FILE_TESTING_LONG) as test_short, open(FILE_TESTING_USER) as test_user:
                test_long_lines = test_short.readlines()
                user_lines = test_user.readlines()
                for i in range(len(user_lines)):
                    test_user = user_lines[i].split('\n')
                    test_long_line = test_long_lines[i].replace('\n','').replace(' ','').replace('[[','').replace(']]','').split('],[')
                    k_user = []; k_lastitem = []
                    k_context = []; k_context_p = []
                    k_cat1 = []; k_cat2 = []; k_cat3 = []; k_cat4 = []; k_cat5 = []; k_cat6 = []; k_cat7 = []
                    for k in range(len(test_long_line)):
                       
                        current_session = test_long_line[k]
                        current_item_session = current_session.split(',')
                        
                        itemId_list = []
                        for l in range(len(current_item_session)):
                            itemId_list.append(current_item_session[l])
                        
                        
                        user = []
                        user.append(user_index.get(int(test_user[0])))
                        lastitem = []
                        lastitem.append(word2idx(itemId_list[-1:][0]))
                        current_session = itemId_list
                        
                        test_long_genre = df_genre.loc[df_genre['ItemId'].isin(current_session)]
                        
                        
                        total_items = len(current_session) #if short we exclude the last item
                    
                        total_community = test_short_genre['genre_Community'].sum()
                        if total_community>max_item_cat1: max_item_cat1= total_community
                        tmp_community_items = list(test_short_genre[test_short_genre['genre_Community'] ==1]['ItemId']) #not consider order of item
                        community_items = [word2idx(str(x)) for x in tmp_community_items]
                        if total_community == 0: community_items.append(0)
                        
                        total_entertainment = test_short_genre['genre_Entertainment'].sum()
                        if total_entertainment>max_item_cat2: max_item_cat2= total_entertainment
                        tmp_entertainment_items = list(test_short_genre[test_short_genre['genre_Entertainment'] ==1]['ItemId']) 
                        entertainment_items = [word2idx(str(x)) for x in tmp_entertainment_items]
                        if total_entertainment == 0: entertainment_items.append(0)
                        
                        total_food = test_short_genre['genre_Food'].sum()
                        if total_food>max_item_cat3: max_item_cat3= total_food
                        tmp_food_items = list(test_short_genre[test_short_genre['genre_Food'] ==1]['ItemId']) 
                        food_items = [word2idx(str(x)) for x in tmp_food_items]
                        if total_food == 0: food_items.append(0)
                        
                        total_nightlife = test_short_genre['genre_Nightlife'].sum()
                        if total_nightlife>max_item_cat4: max_item_cat4= total_nightlife
                        tmp_nightlife_items = list(test_short_genre[test_short_genre['genre_Nightlife'] ==1]['ItemId']) 
                        nightlife_items = [word2idx(str(x)) for x in tmp_nightlife_items]
                        if total_nightlife == 0: nightlife_items.append(0)
                        
                        total_outdoors = test_short_genre['genre_Outdoors'].sum()
                        if total_outdoors>max_item_cat5: max_item_cat5= total_outdoors
                        tmp_outdoors_items = list(test_short_genre[test_short_genre['genre_Outdoors'] ==1]['ItemId']) 
                        outdoors_items = [word2idx(str(x)) for x in tmp_outdoors_items]
                        if total_outdoors == 0: outdoors_items.append(0)
                        
                        total_shopping = test_short_genre['genre_Shopping'].sum()
                        if total_shopping>max_item_cat6: max_item_cat6= total_shopping
                        tmp_shopping_items = list(test_short_genre[test_short_genre['genre_Shopping'] ==1]['ItemId']) 
                        shopping_items = [word2idx(str(x)) for x in tmp_shopping_items]
                        if total_shopping == 0: shopping_items.append(0)
                        
                        total_travel = test_short_genre['genre_Travel'].sum()
                        if total_travel>max_item_cat7: max_item_cat7= total_travel
                        tmp_travel_items = list(test_short_genre[test_short_genre['genre_Travel'] ==1]['ItemId']) 
                        travel_items = [word2idx(str(x)) for x in tmp_travel_items]
                        if total_travel == 0: travel_items.append(0)
                        
                        tmp_context_p = [calculate_class(total_community/total_items), calculate_class(total_entertainment/total_items), calculate_class(total_food/total_items),
                                     calculate_class(total_nightlife/total_items), calculate_class(total_outdoors/total_items), calculate_class(total_shopping/total_items), calculate_class(total_travel/total_items)]
                
                        context_p = [contextpercent_index[x] for x in tmp_context_p]
                    
                        context = [context_index[x] for x in CONTEXTS]
                    
                        k_user.append(user)
                        k_lastitem.append(lastitem)
                        k_context.append(context)
                        k_context_p.append(context_p)
                        k_cat1.append(community_items)
                        k_cat2.append(entertainment_items)
                        k_cat3.append(food_items)
                        k_cat4.append(nightlife_items)
                        k_cat5.append(outdoors_items)
                        k_cat6.append(shopping_items)
                        k_cat7.append(travel_items)
                        
                        
                    test_long_user.append(k_user)
                    test_long_lastitem.append(k_lastitem)
                    test_long_context.append(k_context)
                    test_long_context_p.append(k_context_p)
                    test_long_cat1.append(k_cat1)
                    test_long_cat2.append(k_cat2)
                    test_long_cat3.append(k_cat3)
                    test_long_cat4.append(k_cat4)
                    test_long_cat5.append(k_cat5)
                    test_long_cat6.append(k_cat6)
                    test_long_cat7.append(k_cat7)
            
           
            
            #---------------------------TRAINING_SHORT-----------------------#
            with open(FILE_TRAINING_SHORT) as train_short, open(FILE_TRAINING_USER) as train_user:
                train_short_lines = train_short.readlines()
                user_lines = train_user.readlines()
                for i in range(len(user_lines)):
                    train_user = user_lines[i].split('\n')
                    
                    train_short_line = train_short_lines[i].replace('\n','').replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')
                    itemId_list = []
                    for l in range(len(train_short_line)):
                        itemId_list.append(train_short_line[l])
                    
                    
                    user = []
                    
                    user.append(user_index.get(int(train_user[0])))
                    output = word2idx(itemId_list[-1:][0])
                    last_item = word2idx(itemId_list[-2:-1][0])
                    current_session = itemId_list[:-1] 
                    seqlastsession = [word2idx(str(x)) for x in current_session]
                    
                    train_short_genre = df_genre.loc[df_genre['ItemId'].isin(current_session)]
                    
                    
                    total_items = len(current_session) #if short we exclude the last item
                    
                    total_community = test_short_genre['genre_Community'].sum()
                    if total_community>max_item_cat1: max_item_cat1= total_community
                    tmp_community_items = list(test_short_genre[test_short_genre['genre_Community'] ==1]['ItemId']) #not consider order of item
                    community_items = [word2idx(str(x)) for x in tmp_community_items]
                    if total_community == 0: community_items.append(0)
                    
                    total_entertainment = test_short_genre['genre_Entertainment'].sum()
                    if total_entertainment>max_item_cat2: max_item_cat2= total_entertainment
                    tmp_entertainment_items = list(test_short_genre[test_short_genre['genre_Entertainment'] ==1]['ItemId']) 
                    entertainment_items = [word2idx(str(x)) for x in tmp_entertainment_items]
                    if total_entertainment == 0: entertainment_items.append(0)
                    
                    total_food = test_short_genre['genre_Food'].sum()
                    if total_food>max_item_cat3: max_item_cat3= total_food
                    tmp_food_items = list(test_short_genre[test_short_genre['genre_Food'] ==1]['ItemId']) 
                    food_items = [word2idx(str(x)) for x in tmp_food_items]
                    if total_food == 0: food_items.append(0)
                    
                    total_nightlife = test_short_genre['genre_Nightlife'].sum()
                    if total_nightlife>max_item_cat4: max_item_cat4= total_nightlife
                    tmp_nightlife_items = list(test_short_genre[test_short_genre['genre_Nightlife'] ==1]['ItemId']) 
                    nightlife_items = [word2idx(str(x)) for x in tmp_nightlife_items]
                    if total_nightlife == 0: nightlife_items.append(0)
                    
                    total_outdoors = test_short_genre['genre_Outdoors'].sum()
                    if total_outdoors>max_item_cat5: max_item_cat5= total_outdoors
                    tmp_outdoors_items = list(test_short_genre[test_short_genre['genre_Outdoors'] ==1]['ItemId']) 
                    outdoors_items = [word2idx(str(x)) for x in tmp_outdoors_items]
                    if total_outdoors == 0: outdoors_items.append(0)
                    
                    total_shopping = test_short_genre['genre_Shopping'].sum()
                    if total_shopping>max_item_cat6: max_item_cat6= total_shopping
                    tmp_shopping_items = list(test_short_genre[test_short_genre['genre_Shopping'] ==1]['ItemId']) 
                    shopping_items = [word2idx(str(x)) for x in tmp_shopping_items]
                    if total_shopping == 0: shopping_items.append(0)
                    
                    total_travel = test_short_genre['genre_Travel'].sum()
                    if total_travel>max_item_cat7: max_item_cat7= total_travel
                    tmp_travel_items = list(test_short_genre[test_short_genre['genre_Travel'] ==1]['ItemId']) 
                    travel_items = [word2idx(str(x)) for x in tmp_travel_items]
                    if total_travel == 0: travel_items.append(0)
                    
                    tmp_context_p = [calculate_class(total_community/total_items), calculate_class(total_entertainment/total_items), calculate_class(total_food/total_items),
                                 calculate_class(total_nightlife/total_items), calculate_class(total_outdoors/total_items), calculate_class(total_shopping/total_items), calculate_class(total_travel/total_items)]
            
                    context_p = [contextpercent_index[x] for x in tmp_context_p]
                    context = [context_index[x] for x in CONTEXTS]
                    
                    train_short_user.append(user)
                    train_short_output.append(output)
                    train_short_lastitem.append(lastitem)
                    train_short_seqlastsession.append(seqlastsession)
                    train_short_context.append(context)
                    train_short_context_p.append(context_p)
                    train_short_cat1.append(community_items)
                    train_short_cat2.append(entertainment_items)
                    train_short_cat3.append(food_items)
                    train_short_cat4.append(nightlife_items)
                    train_short_cat5.append(outdoors_items)
                    train_short_cat6.append(shopping_items)
                    train_short_cat7.append(travel_items)
 
            #-----------------------TRAINING_LONG-----------------------#
            with open(FILE_TRAINING_LONG) as train_long, open(FILE_TRAINING_USER) as train_user:
                train_long_lines = train_long.readlines()
                user_lines = train_user.readlines()
                for i in range(len(user_lines)):
                    train_user = user_lines[i].split('\n')
                    train_long_line = train_long_lines[i].replace('\n','').replace(' ','').replace('[[','').replace(']]','').split('],[')
                    
                    k_user = []; k_lastitem = []
                    k_context = []; k_context_p = []
                    k_cat1 = []; k_cat2 = []; k_cat3 = []; k_cat4 = []; k_cat5 = []; k_cat6 = []; k_cat7 = []
                    for k in range(len(train_long_line)):
                        
                        current_session = train_long_line[k]
                        current_item_session = current_session.split(',')
                        
                        itemId_list = []
                        for l in range(len(current_item_session)):
                            itemId_list.append(current_item_session[l])
                        
                        
                        user = []
                        user.append(user_index.get(int(train_user[0])))
                        lastitem = []
                        lastitem.append(word2idx(itemId_list[-1:][0]))
                        current_session = itemId_list
                        
                        
                        train_long_genre = df_genre.loc[df_genre['ItemId'].isin(current_session)]
                       
                        
                        total_items = len(current_session) #if short we exclude the last item
                    
                        total_community = test_short_genre['genre_Community'].sum()
                        if total_community>max_item_cat1: max_item_cat1= total_community
                        tmp_community_items = list(test_short_genre[test_short_genre['genre_Community'] ==1]['ItemId']) #not consider order of item
                        community_items = [word2idx(str(x)) for x in tmp_community_items]
                        if total_community == 0: community_items.append(0)
                        
                        total_entertainment = test_short_genre['genre_Entertainment'].sum()
                        if total_entertainment>max_item_cat2: max_item_cat2= total_entertainment
                        tmp_entertainment_items = list(test_short_genre[test_short_genre['genre_Entertainment'] ==1]['ItemId']) 
                        entertainment_items = [word2idx(str(x)) for x in tmp_entertainment_items]
                        if total_entertainment == 0: entertainment_items.append(0)
                        
                        total_food = test_short_genre['genre_Food'].sum()
                        if total_food>max_item_cat3: max_item_cat3= total_food
                        tmp_food_items = list(test_short_genre[test_short_genre['genre_Food'] ==1]['ItemId']) 
                        food_items = [word2idx(str(x)) for x in tmp_food_items]
                        if total_food == 0: food_items.append(0)
                        
                        total_nightlife = test_short_genre['genre_Nightlife'].sum()
                        if total_nightlife>max_item_cat4: max_item_cat4= total_nightlife
                        tmp_nightlife_items = list(test_short_genre[test_short_genre['genre_Nightlife'] ==1]['ItemId']) 
                        nightlife_items = [word2idx(str(x)) for x in tmp_nightlife_items]
                        if total_nightlife == 0: nightlife_items.append(0)
                        
                        total_outdoors = test_short_genre['genre_Outdoors'].sum()
                        if total_outdoors>max_item_cat5: max_item_cat5= total_outdoors
                        tmp_outdoors_items = list(test_short_genre[test_short_genre['genre_Outdoors'] ==1]['ItemId']) 
                        outdoors_items = [word2idx(str(x)) for x in tmp_outdoors_items]
                        if total_outdoors == 0: outdoors_items.append(0)
                        
                        total_shopping = test_short_genre['genre_Shopping'].sum()
                        if total_shopping>max_item_cat6: max_item_cat6= total_shopping
                        tmp_shopping_items = list(test_short_genre[test_short_genre['genre_Shopping'] ==1]['ItemId']) 
                        shopping_items = [word2idx(str(x)) for x in tmp_shopping_items]
                        if total_shopping == 0: shopping_items.append(0)
                        
                        total_travel = test_short_genre['genre_Travel'].sum()
                        if total_travel>max_item_cat7: max_item_cat7= total_travel
                        tmp_travel_items = list(test_short_genre[test_short_genre['genre_Travel'] ==1]['ItemId']) 
                        travel_items = [word2idx(str(x)) for x in tmp_travel_items]
                        if total_travel == 0: travel_items.append(0)
                        
                        tmp_context_p = [calculate_class(total_community/total_items), calculate_class(total_entertainment/total_items), calculate_class(total_food/total_items),
                                     calculate_class(total_nightlife/total_items), calculate_class(total_outdoors/total_items), calculate_class(total_shopping/total_items), calculate_class(total_travel/total_items)]
                
                        context_p = [contextpercent_index[x] for x in tmp_context_p]
                    
                        context = [context_index[x] for x in CONTEXTS]
                        
                        k_user.append(user)
                        k_lastitem.append(lastitem)
                        k_context.append(context)
                        k_context_p.append(context_p)
                        k_cat1.append(community_items)
                        k_cat2.append(entertainment_items)
                        k_cat3.append(food_items)
                        k_cat4.append(nightlife_items)
                        k_cat5.append(outdoors_items)
                        k_cat6.append(shopping_items)
                        k_cat7.append(travel_items)
                        
                    train_long_user.append(k_user)
                    train_long_lastitem.append(k_lastitem)
                    train_long_context.append(k_context)
                    train_long_context_p.append(k_context_p)
                    train_long_cat1.append(k_cat1)
                    train_long_cat2.append(k_cat2)
                    train_long_cat3.append(k_cat3)
                    train_long_cat4.append(k_cat4)
                    train_long_cat5.append(k_cat5)
                    train_long_cat6.append(k_cat6)
                    train_long_cat7.append(k_cat7)
            
            #-------------------------Change Type----------------------# 
            train_short_user = np.asarray(train_short_user)
            train_short_output = np.asarray(train_short_output)
            train_short_lastitem = np.asarray(train_short_lastitem)
            train_short_seqlastsession = np.asarray(train_short_seqlastsession)
            train_short_context = np.asarray(train_short_context)
            train_short_context_p = np.asarray(train_short_context_p)
            train_short_cat1 = np.asarray(train_short_cat1)
            train_short_cat2= np.asarray(train_short_cat2)
            train_short_cat3= np.asarray(train_short_cat3)
            train_short_cat4= np.asarray(train_short_cat4)
            train_short_cat5= np.asarray(train_short_cat5)
            train_short_cat6= np.asarray(train_short_cat6)
            train_short_cat7= np.asarray(train_short_cat7)
            
            train_long_user = np.asarray(train_long_user)
            train_long_lastitem = np.asarray(train_long_lastitem)
            train_long_context = np.asarray(train_long_context)
            train_long_context_p = np.asarray(train_long_context_p)
            train_long_cat1 = np.asarray(train_long_cat1)
            train_long_cat2 = np.asarray(train_long_cat2)
            train_long_cat3 = np.asarray(train_long_cat3)
            train_long_cat4 = np.asarray(train_long_cat4)
            train_long_cat5 = np.asarray(train_long_cat5)
            train_long_cat6 = np.asarray(train_long_cat6)
            train_long_cat7 = np.asarray(train_long_cat7)
            
            test_short_user = np.asarray(test_short_user)
            test_short_output = np.asarray(test_short_output)
            test_short_lastitem = np.asarray(test_short_lastitem)
            test_short_seqlastsession = np.asarray(test_short_seqlastsession)
            test_short_context = np.asarray(test_short_context)
            test_short_context_p = np.asarray(test_short_context_p)
            test_short_cat1= np.asarray(test_short_cat1)
            test_short_cat2= np.asarray(test_short_cat2)
            test_short_cat3= np.asarray(test_short_cat3)
            test_short_cat4= np.asarray(test_short_cat4)
            test_short_cat5= np.asarray(test_short_cat5)
            test_short_cat6= np.asarray(test_short_cat6)
            test_short_cat7= np.asarray(test_short_cat7)
            
            test_long_user = np.asarray(test_long_user)
            test_long_lastitem = np.asarray(test_long_lastitem)
            test_long_context = np.asarray(test_long_context)
            test_long_context_p = np.asarray(test_long_context_p)
            test_long_cat1= np.asarray(test_long_cat1)
            test_long_cat2= np.asarray(test_long_cat2)
            test_long_cat3= np.asarray(test_long_cat3)
            test_long_cat4= np.asarray(test_long_cat4)
            test_long_cat5= np.asarray(test_long_cat5)
            test_long_cat6= np.asarray(test_long_cat6)
            test_long_cat7= np.asarray(test_long_cat7)
            
           
            #--------------------cat1-------------------#
            train_short_cat1 = pad_sequences(train_short_cat1,padding='pre' ,maxlen=max_item_cat1)
            test_short_cat1 = pad_sequences(test_short_cat1,padding='pre' ,maxlen=max_item_cat1)
            tmp = []
            for i in range(train_long_cat1.shape[0]):
                tmp.append(pad_sequences(train_long_cat1[i],padding='pre' ,maxlen=max_item_cat1))
            train_long_cat1 = tmp
            train_long_cat1 = np.asarray(train_long_cat1)
            tmp = []
            for i in range(test_long_cat1.shape[0]):
                tmp.append(pad_sequences(test_long_cat1[i],padding='pre' ,maxlen=max_item_cat1))
            test_long_cat1 = tmp
            test_long_cat1 = np.asarray(test_long_cat1)
            

            #--------------------cat2-------------------#
            train_short_cat2 = pad_sequences(train_short_cat2,padding='pre' ,maxlen=max_item_cat2)
            test_short_cat2 = pad_sequences(test_short_cat2,padding='pre' ,maxlen=max_item_cat2)
            tmp = []
            for i in range(train_long_cat2.shape[0]):
                tmp.append(pad_sequences(train_long_cat2[i],padding='pre' ,maxlen=max_item_cat2))
            train_long_cat2 = tmp
            train_long_cat2 = np.asarray(train_long_cat2)
            tmp = []
            for i in range(test_long_cat2.shape[0]):
                tmp.append(pad_sequences(test_long_cat2[i],padding='pre' ,maxlen=max_item_cat2))
            test_long_cat2 = tmp
            test_long_cat2 = np.asarray(test_long_cat2)
            
   
            
            #--------------------cat3-------------------#
            train_short_cat3 = pad_sequences(train_short_cat3,padding='pre' ,maxlen=max_item_cat3)
            test_short_cat3 = pad_sequences(test_short_cat3,padding='pre' ,maxlen=max_item_cat3)
            tmp = []
            for i in range(train_long_cat3.shape[0]):
                tmp.append(pad_sequences(train_long_cat3[i],padding='pre' ,maxlen=max_item_cat3))
            train_long_cat3 = tmp
            train_long_cat3 = np.asarray(train_long_cat3)
            tmp = []
            for i in range(test_long_cat3.shape[0]):
                tmp.append(pad_sequences(test_long_cat3[i],padding='pre' ,maxlen=max_item_cat3))
            test_long_cat3 = tmp
            test_long_cat3 = np.asarray(test_long_cat3)

            #--------------------cat4-------------------#
            train_short_cat4 = pad_sequences(train_short_cat4,padding='pre' ,maxlen=max_item_cat4)
            test_short_cat4 = pad_sequences(test_short_cat4,padding='pre' ,maxlen=max_item_cat4)
            tmp = []
            for i in range(train_long_cat4.shape[0]):
                tmp.append(pad_sequences(train_long_cat4[i],padding='pre' ,maxlen=max_item_cat4))
            train_long_cat4 = tmp
            train_long_cat4 = np.asarray(train_long_cat4)
            tmp = []
            for i in range(test_long_cat4.shape[0]):
                tmp.append(pad_sequences(test_long_cat4[i],padding='pre' ,maxlen=max_item_cat4))
            test_long_cat4 = tmp
            test_long_cat4 = np.asarray(test_long_cat4)

            #--------------------cat5-------------------#
            train_short_cat5 = pad_sequences(train_short_cat5,padding='pre' ,maxlen=max_item_cat5)
            test_short_cat5 = pad_sequences(test_short_cat5,padding='pre' ,maxlen=max_item_cat5)
            tmp = []
            for i in range(train_long_cat5.shape[0]):
                tmp.append(pad_sequences(train_long_cat5[i],padding='pre' ,maxlen=max_item_cat5))
            train_long_cat5 = tmp
            train_long_cat5 = np.asarray(train_long_cat5)
            tmp = []
            for i in range(test_long_cat5.shape[0]):
                tmp.append(pad_sequences(test_long_cat5[i],padding='pre' ,maxlen=max_item_cat5))
            test_long_cat5 = tmp
            test_long_cat5 = np.asarray(test_long_cat5)
            
 
            
            #--------------------cat6-------------------#
            train_short_cat6 = pad_sequences(train_short_cat6,padding='pre' ,maxlen=max_item_cat6)
            test_short_cat6 = pad_sequences(test_short_cat6,padding='pre' ,maxlen=max_item_cat6)
            tmp = []
            for i in range(train_long_cat6.shape[0]):
                tmp.append(pad_sequences(train_long_cat6[i],padding='pre' ,maxlen=max_item_cat6))
            train_long_cat6 = tmp
            train_long_cat6 = np.asarray(train_long_cat6)
            tmp = []
            for i in range(test_long_cat6.shape[0]):
                tmp.append(pad_sequences(test_long_cat6[i],padding='pre' ,maxlen=max_item_cat6))
            test_long_cat6 = tmp
            test_long_cat6 = np.asarray(test_long_cat6)
            
 
            #--------------------cat7-------------------#
            train_short_cat7 = pad_sequences(train_short_cat7,padding='pre' ,maxlen=max_item_cat7)
            test_short_cat7 = pad_sequences(test_short_cat7,padding='pre' ,maxlen=max_item_cat7)
            tmp = []
            for i in range(train_long_cat7.shape[0]):
                tmp.append(pad_sequences(train_long_cat7[i],padding='pre' ,maxlen=max_item_cat7))
            train_long_cat7 = tmp
            train_long_cat7 = np.asarray(train_long_cat7)
            tmp = []
            for i in range(test_long_cat7.shape[0]):
                tmp.append(pad_sequences(test_long_cat7[i],padding='pre' ,maxlen=max_item_cat7))
            test_long_cat7 = tmp
            test_long_cat7 = np.asarray(test_long_cat7)
            
 
            
            #--------------------seqoflastsession-------------------#
            train_short_seqlastsession = pad_sequences(train_short_seqlastsession,padding='pre' ,maxlen=n)  #50
            test_short_seqlastsession = pad_sequences(test_short_seqlastsession,padding='pre' ,maxlen=n)    #50
            
            #---------------------Write to file--------------------# 
            FILE_TRAIN_SHORT_USER = 'datasets/gw_train/short_user.txt'
            FILE_TRAIN_ITEMOUTPUT = 'datasets/gw_train/itemoutput.txt'
            FILE_TRAIN_SHORT_LASTITEM = 'datasets/gw_train/lastitem.txt'
            FILE_TRAIN_SHORT_SEQLASTSESSION = 'datasets/gw_train/seqlastsession.txt'
            
            FILE_TRAIN_SHORT_CONTEXT = 'datasets/gw_train/short_context.txt'
            FILE_TRAIN_SHORT_CONTEXT_P = 'datasets/gw_train/short_context_p.txt'
            FILE_TRAIN_SHORT_CAT1 = 'datasets/gw_train/short_itemcat1.txt'
            FILE_TRAIN_SHORT_CAT2 = 'datasets/gw_train/short_itemcat2.txt'
            FILE_TRAIN_SHORT_CAT3 = 'datasets/gw_train/short_itemcat3.txt'
            FILE_TRAIN_SHORT_CAT4 = 'datasets/gw_train/short_itemcat4.txt'
            FILE_TRAIN_SHORT_CAT5 = 'datasets/gw_train/short_itemcat5.txt'
            FILE_TRAIN_SHORT_CAT6 = 'datasets/gw_train/short_itemcat6.txt'
            FILE_TRAIN_SHORT_CAT7 = 'datasets/gw_train/short_itemcat7.txt'
            
            FILE_TRAIN_LONG_USER = 'datasets/gw_train/long_user.txt'
            FILE_TRAIN_LONG_LASTITEM = 'datasets/gw_train/long_lastitem.txt'
            
            FILE_TRAIN_LONG_CONTEXT = 'datasets/gw_train/long_context.txt'
            FILE_TRAIN_LONG_CONTEXT_P = 'datasets/gw_train/long_context_p.txt'
            FILE_TRAIN_LONG_CAT1 = 'datasets/gw_train/long_itemcat1.txt'
            FILE_TRAIN_LONG_CAT2 = 'datasets/gw_train/long_itemcat2.txt'
            FILE_TRAIN_LONG_CAT3 = 'datasets/gw_train/long_itemcat3.txt'
            FILE_TRAIN_LONG_CAT4 = 'datasets/gw_train/long_itemcat4.txt'
            FILE_TRAIN_LONG_CAT5 = 'datasets/gw_train/long_itemcat5.txt'
            FILE_TRAIN_LONG_CAT6 = 'datasets/gw_train/long_itemcat6.txt'
            FILE_TRAIN_LONG_CAT7 = 'datasets/gw_train/long_itemcat7.txt'
            
            FILE_TEST_SHORT_USER = 'datasets/gw_test/short_user.txt'
            FILE_TEST_ITEMOUTPUT = 'datasets/gw_test/itemoutput.txt'
            FILE_TEST_SHORT_LASTITEM = 'datasets/gw_test/lastitem.txt'
            FILE_TEST_SHORT_SEQLASTSESSION = 'datasets/gw_test/seqlastsession.txt'
            
            FILE_TEST_SHORT_CONTEXT = 'datasets/gw_test/short_context.txt'
            FILE_TEST_SHORT_CONTEXT_P = 'datasets/gw_test/short_context_p.txt'
            FILE_TEST_SHORT_CAT1 = 'datasets/gw_test/short_itemcat1.txt'
            FILE_TEST_SHORT_CAT2 = 'datasets/gw_test/short_itemcat2.txt'
            FILE_TEST_SHORT_CAT3 = 'datasets/gw_test/short_itemcat3.txt'
            FILE_TEST_SHORT_CAT4 = 'datasets/gw_test/short_itemcat4.txt'
            FILE_TEST_SHORT_CAT5 = 'datasets/gw_test/short_itemcat5.txt'
            FILE_TEST_SHORT_CAT6 = 'datasets/gw_test/short_itemcat6.txt'
            FILE_TEST_SHORT_CAT7 = 'datasets/gw_test/short_itemcat7.txt'
            
            FILE_TEST_LONG_USER = 'datasets/gw_test/long_user.txt'
            FILE_TEST_LONG_LASTITEM = 'datasets/gw_test/long_lastitem.txt'
            
            FILE_TEST_LONG_CONTEXT = 'datasets/gw_test/long_context.txt'
            FILE_TEST_LONG_CONTEXT_P = 'datasets/gw_test/long_context_p.txt'
            FILE_TEST_LONG_CAT1 = 'datasets/gw_test/long_itemcat1.txt'
            FILE_TEST_LONG_CAT2 = 'datasets/gw_test/long_itemcat2.txt'
            FILE_TEST_LONG_CAT3 = 'datasets/gw_test/long_itemcat3.txt'
            FILE_TEST_LONG_CAT4 = 'datasets/gw_test/long_itemcat4.txt'
            FILE_TEST_LONG_CAT5 = 'datasets/gw_test/long_itemcat5.txt'
            FILE_TEST_LONG_CAT6 = 'datasets/gw_test/long_itemcat6.txt'
            FILE_TEST_LONG_CAT7 = 'datasets/gw_test/long_itemcat7.txt'
            
            #-----------Save numpyarray-----------#
           
            with open(FILE_TRAIN_SHORT_USER, 'wb') as fp:
                pickle.dump(train_short_user, fp)
            #np.save(FILE_TRAIN_SHORT_USER, train_short_user)
            with open(FILE_TRAIN_ITEMOUTPUT, 'wb') as fp:
                pickle.dump(train_short_output, fp)
            with open(FILE_TRAIN_SHORT_LASTITEM, 'wb') as fp:
                pickle.dump(train_short_lastitem, fp)
            with open(FILE_TRAIN_SHORT_SEQLASTSESSION, 'wb') as fp:
                pickle.dump(train_short_seqlastsession, fp)
            with open(FILE_TRAIN_SHORT_CONTEXT, 'wb') as fp:
                pickle.dump(train_short_context, fp)
            with open(FILE_TRAIN_SHORT_CONTEXT_P, 'wb') as fp:
                pickle.dump(train_short_context_p, fp)
            with open(FILE_TRAIN_SHORT_CAT1, 'wb') as fp:
                pickle.dump(train_short_cat1, fp)
            with open(FILE_TRAIN_SHORT_CAT2, 'wb') as fp:
                pickle.dump(train_short_cat2, fp)
            with open(FILE_TRAIN_SHORT_CAT3, 'wb') as fp:
                pickle.dump(train_short_cat3, fp)
            with open(FILE_TRAIN_SHORT_CAT4, 'wb') as fp:
                pickle.dump(train_short_cat4, fp)
            with open(FILE_TRAIN_SHORT_CAT5, 'wb') as fp:
                pickle.dump(train_short_cat5, fp)
            with open(FILE_TRAIN_SHORT_CAT6, 'wb') as fp:
                pickle.dump(train_short_cat6, fp)
            with open(FILE_TRAIN_SHORT_CAT7, 'wb') as fp:
                pickle.dump(train_short_cat7, fp)
            
            with open(FILE_TRAIN_LONG_USER, 'wb') as fp:
                pickle.dump(train_long_user, fp)
            with open(FILE_TRAIN_LONG_LASTITEM, 'wb') as fp:
                pickle.dump(train_long_lastitem, fp)
            with open(FILE_TRAIN_LONG_CONTEXT, 'wb') as fp:
                pickle.dump(train_long_context, fp)
            with open(FILE_TRAIN_LONG_CONTEXT_P, 'wb') as fp:
                pickle.dump(train_long_context_p, fp)
            with open(FILE_TRAIN_LONG_CAT1, 'wb') as fp:
                pickle.dump(train_long_cat1, fp)
            with open(FILE_TRAIN_LONG_CAT2, 'wb') as fp:
                pickle.dump(train_long_cat2, fp)
            with open(FILE_TRAIN_LONG_CAT3, 'wb') as fp:
                pickle.dump(train_long_cat3, fp)
            with open(FILE_TRAIN_LONG_CAT4, 'wb') as fp:
                pickle.dump(train_long_cat4, fp)
            with open(FILE_TRAIN_LONG_CAT5, 'wb') as fp:
                pickle.dump(train_long_cat5, fp)
            with open(FILE_TRAIN_LONG_CAT6, 'wb') as fp:
                pickle.dump(train_long_cat6, fp)
            with open(FILE_TRAIN_LONG_CAT7, 'wb') as fp:
                pickle.dump(train_long_cat7, fp)
                
            with open(FILE_TEST_SHORT_USER, 'wb') as fp:
                pickle.dump(test_short_user, fp)
            with open(FILE_TEST_ITEMOUTPUT, 'wb') as fp:
                pickle.dump(test_short_output, fp)
            with open(FILE_TEST_SHORT_LASTITEM, 'wb') as fp:
                pickle.dump(test_short_lastitem, fp)
            with open(FILE_TEST_SHORT_SEQLASTSESSION, 'wb') as fp:
                pickle.dump(test_short_seqlastsession, fp)
            with open(FILE_TEST_SHORT_CONTEXT, 'wb') as fp:
                pickle.dump(test_short_context, fp)
            with open(FILE_TEST_SHORT_CONTEXT_P, 'wb') as fp:
                pickle.dump(test_short_context_p, fp)
            with open(FILE_TEST_SHORT_CAT1, 'wb') as fp:
                pickle.dump(test_short_cat1, fp)
            with open(FILE_TEST_SHORT_CAT2, 'wb') as fp:
                pickle.dump(test_short_cat2, fp)
            with open(FILE_TEST_SHORT_CAT3, 'wb') as fp:
                pickle.dump(test_short_cat3, fp)
            with open(FILE_TEST_SHORT_CAT4, 'wb') as fp:
                pickle.dump(test_short_cat4, fp)
            with open(FILE_TEST_SHORT_CAT5, 'wb') as fp:
                pickle.dump(test_short_cat5, fp)
            with open(FILE_TEST_SHORT_CAT6, 'wb') as fp:
                pickle.dump(test_short_cat6, fp)
            with open(FILE_TEST_SHORT_CAT7, 'wb') as fp:
                pickle.dump(test_short_cat7, fp)
            
            with open(FILE_TEST_LONG_USER, 'wb') as fp:
                pickle.dump(test_long_user, fp)
            with open(FILE_TEST_LONG_LASTITEM, 'wb') as fp:
                pickle.dump(test_long_lastitem, fp)
            with open(FILE_TEST_LONG_CONTEXT, 'wb') as fp:
                pickle.dump(test_long_context, fp)
            with open(FILE_TEST_LONG_CONTEXT_P, 'wb') as fp:
                pickle.dump(test_long_context_p, fp)
            with open(FILE_TEST_LONG_CAT1, 'wb') as fp:
                pickle.dump(test_long_cat1, fp)
            with open(FILE_TEST_LONG_CAT2, 'wb') as fp:
                pickle.dump(test_long_cat2, fp)
            with open(FILE_TEST_LONG_CAT3, 'wb') as fp:
                pickle.dump(test_long_cat3, fp)
            with open(FILE_TEST_LONG_CAT4, 'wb') as fp:
                pickle.dump(test_long_cat4, fp)
            with open(FILE_TEST_LONG_CAT5, 'wb') as fp:
                pickle.dump(test_long_cat5, fp)
            with open(FILE_TEST_LONG_CAT6, 'wb') as fp:
                pickle.dump(test_long_cat6, fp)
            with open(FILE_TEST_LONG_CAT7, 'wb') as fp:
                pickle.dump(test_long_cat7, fp)
            
            with open('datasets/gw_maxitem.txt', "w") as text_file:
                    print('max_item_cat1:{}\nmax_item_cat2:{}\nmax_item_cat3:{} \nmax_item_cat4:{}  \nmax_item_cat5:{} \nmax_item_cat6:{} \nmax_item_cat7:{}'.
                          format( max_item_cat1, max_item_cat2, max_item_cat3, max_item_cat4, 
                         max_item_cat5, max_item_cat6, max_item_cat7 ), file=text_file)
                   
