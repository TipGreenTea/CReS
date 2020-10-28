"""
CReS GW
"""
import tensorflow as tf
from keras import backend as K
from keras_preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate, Reshape, Multiply, Add, Lambda, Masking, Embedding, TimeDistributed
from tensorflow.python.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from keras import initializers
import numpy as np
from numpy import savetxt, loadtxt
from numpy import argmax
import gensim 
from gensim.models import Word2Vec
import pandas as pd 
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib import pyplot
from statistics import mean
"""
files & load all parameters & load all dictionaries
"""
MIN_ITEM_SUPPORT = [5] 
MAX_SESSION_ITEM = [50] 
K_HIS = [10] 
TOTAL_CONTEXT = 7
TOTAL_PERCENT = 5
N =20
MODEL_TRAIN_EPOCH = 12

for m in MIN_ITEM_SUPPORT:
    for n in MAX_SESSION_ITEM:
        for k_his in K_HIS:
            
            #------change according to datasets-------#
            FILE_STAT = 'datasets/gw_poi_dataset_filter_stat.txt'
            FILE_MAXITEM = 'datasets/gw_maxitem.txt'
            stat = open(FILE_STAT, "r"); stat_lines = (stat.readlines())
            TOTAL_USER = int(stat_lines[3].replace(" ","").replace("\n","").split(":")[1]) #TOTAL_USER = 1854
            TOTAL_ITEM = int(stat_lines[4].replace(" ","").replace("\n","").split(":")[1])  #TOTAL_ITEM = 4890
            maxitem = open(FILE_MAXITEM, "r"); maxitem_lines = (maxitem.readlines())
            MAX_ITEM_CAT1 = int(maxitem_lines[0].replace(" ","").replace("\n","").split(":")[1]) #16
            MAX_ITEM_CAT2 = int(maxitem_lines[1].replace(" ","").replace("\n","").split(":")[1]) #19
            MAX_ITEM_CAT3 = int(maxitem_lines[2].replace(" ","").replace("\n","").split(":")[1]) #14
            MAX_ITEM_CAT4 = int(maxitem_lines[3].replace(" ","").replace("\n","").split(":")[1]) #8
            MAX_ITEM_CAT5 = int(maxitem_lines[4].replace(" ","").replace("\n","").split(":")[1]) #19
            MAX_ITEM_CAT6 = int(maxitem_lines[5].replace(" ","").replace("\n","").split(":")[1]) #5
            MAX_ITEM_CAT7 = int(maxitem_lines[6].replace(" ","").replace("\n","").split(":")[1]) #8
           
            #-----------Model Parameters----------#
            INPUT_DIM = 32
            N_session = k_his  #K 
            MODEL_BATCH = 64
            SHORT_ATTENTION_UNIT = 20
            LONG_ATTENTION_UNIT = 20
            TEMPORAL_ATTENTION_UNIT = 20
            FINAL_DENSE_UNIT = TOTAL_ITEM
            FINAL_DENSE_ACTIVATION ='relu'
            RNN_CELL_UNIT = 16
            RNN_CELL_DROPOUT = 0#0.3
            RNN_CELL_ACTIVATION = 'relu'
            RNN_CELL_INITIALIZER = 'glorot_uniform'
            
            #----------Embedding Files-----------#
            EMBEDDING_ITEM_NAME = 'datasets/gw_item_embedding'
            USER_EMBEDDING = 'datasets/gw_user_embedding.csv'
            CONTEXT_EMBEDDING = 'datasets/gw_context_embedding.csv'
            CONTEXT_PERCENT_EMBEDDING = 'datasets/gw_contextpercent_embedding.csv'
            USER_INDEX = 'datasets/gw_user_index_dict.txt'
            INDEX_USER = 'datasets/gw_index_user_dict.txt'
            CONTEXT_INDEX = 'datasets/gw_context_index_dict.txt'
            INDEX_CONTEXT = 'datasets/gw_index_context_dict.txt'
            CONTEXTPERCENT_INDEX = 'datasets/gw_contextpercent_index_dict.txt'
            INDEX_CONTEXTPERCENT = 'datasets/gw_index_contextpercent_dict.txt'
            
            #----------Files for Train-----------#
            FILE_TRAIN_ROOT = 'datasets/gw_train/'
            FILE_SHORT_USER = FILE_TRAIN_ROOT+'short_user.txt'
            FILE_SHORT_LASTITEM = FILE_TRAIN_ROOT+'lastitem.txt'
            FILE_SHORT_SEQLASTSESSION = FILE_TRAIN_ROOT+'seqlastsession.txt'
            FILE_SHORT_CONTEXT = FILE_TRAIN_ROOT+'short_context.txt'
            FILE_SHORT_CONTEXT_P = FILE_TRAIN_ROOT+'short_context_p.txt'
            FILE_SHORT_CAT1 = FILE_TRAIN_ROOT+'short_itemcat1.txt'
            FILE_SHORT_CAT2 = FILE_TRAIN_ROOT+'short_itemcat2.txt'
            FILE_SHORT_CAT3 = FILE_TRAIN_ROOT+'short_itemcat3.txt'
            FILE_SHORT_CAT4 = FILE_TRAIN_ROOT+'short_itemcat4.txt'
            FILE_SHORT_CAT5 = FILE_TRAIN_ROOT+'short_itemcat5.txt'
            FILE_SHORT_CAT6 = FILE_TRAIN_ROOT+'short_itemcat6.txt'
            FILE_SHORT_CAT7 = FILE_TRAIN_ROOT+'short_itemcat7.txt'
            FILE_LONG_USER = FILE_TRAIN_ROOT+'long_user.txt'
            FILE_LONG_LASTITEM = FILE_TRAIN_ROOT+'long_lastitem.txt'
            FILE_LONG_CONTEXT = FILE_TRAIN_ROOT+'long_context.txt'
            FILE_LONG_CONTEXT_P = FILE_TRAIN_ROOT+'long_context_p.txt'
            FILE_LONG_CAT1 = FILE_TRAIN_ROOT+'long_itemcat1.txt'
            FILE_LONG_CAT2 = FILE_TRAIN_ROOT+'long_itemcat2.txt'
            FILE_LONG_CAT3 = FILE_TRAIN_ROOT+'long_itemcat3.txt'
            FILE_LONG_CAT4 = FILE_TRAIN_ROOT+'long_itemcat4.txt'
            FILE_LONG_CAT5 = FILE_TRAIN_ROOT+'long_itemcat5.txt'
            FILE_LONG_CAT6 = FILE_TRAIN_ROOT+'long_itemcat6.txt'
            FILE_LONG_CAT7 = FILE_TRAIN_ROOT+'long_itemcat7.txt'
            FILE_OUTPUT = FILE_TRAIN_ROOT+'itemoutput.txt'
            
            #----------Files for Output from Training-----------#
            FILE_MODEL_PLOT = 'datasets/gw_train/model_plot.png'
            FILE_MODEL_LOSSPNG = 'datasets/gw_train/model_loss.png'
            FILE_MODEL_LOSS = 'datasets/gw_train/loss.txt'
            FILE_MODEL_ACC = 'datasets/gw_train/acc.txt'
            FILE_MODEL_VALLOSS = 'datasets/gw_train/valloss.txt'
            FILE_MODEL_VALACC = 'datasets/gw_train/valacc.txt'
            FILE_MODEL_SAVE = 'datasets/gw_train/gw_CReS'
            
            
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
            
            #----------pretrained_weights_padded for deep learning model----------#
            #already pad the 0 embedding from data-preprocessing
            embedding_model = gensim.models.Word2Vec.load(EMBEDDING_ITEM_NAME)
            pretrained_weights = embedding_model.wv.vectors
            vocab_size, emdedding_test_size = pretrained_weights.shape
            pretrained_weights_padded =np.vstack([np.zeros((INPUT_DIM)), pretrained_weights])
            vocab_size_padded = vocab_size +1
            pretrained_weights_padded = np.asarray(pretrained_weights_padded, dtype=np.float32)
            #****have to be used for every models****#
            def word2idx(word):
                return (embedding_model.wv.vocab[word].index) +1    #don't have index =0 
            def idx2word(idx):
                return embedding_model.wv.index2word[idx-1]
            
            user_embedding = loadtxt(USER_EMBEDDING, delimiter=',') 
            context_embedding = loadtxt(CONTEXT_EMBEDDING, delimiter=',') 
            context_percent_embedding = loadtxt(CONTEXT_PERCENT_EMBEDDING, delimiter=',')   
            context_index = load_dict_from_file(CONTEXT_INDEX)
            index_context = load_dict_from_file(INDEX_CONTEXT)
            contextpercent_index = load_dict_from_file(CONTEXTPERCENT_INDEX)
            index_contextpercent = load_dict_from_file(INDEX_CONTEXTPERCENT)
            user_index = load_dict_from_file(USER_INDEX)
            index_user = load_dict_from_file(INDEX_USER)
            
            """
            Prepare Input & Output Data for Training
            """
            #-----------Load All Variables-----------#
            #short_user_input= []
            with open (FILE_SHORT_USER, 'rb') as fp: short_user_input = pickle.load(fp)
            short_user_input = np.asarray(short_user_input)
            #print("short_user_input: ", short_user_input)
            with open (FILE_LONG_USER, 'rb') as fp: long_user_input = pickle.load(fp)
            long_user_input = np.asarray(long_user_input)
            with open (FILE_SHORT_CONTEXT, 'rb') as fp: short_context_input = pickle.load(fp)
            short_context_input = np.asarray(short_context_input)
            with open (FILE_LONG_CONTEXT, 'rb') as fp: long_context_input = pickle.load(fp)
            long_context_input = np.asarray(long_context_input)
            with open (FILE_SHORT_CONTEXT_P, 'rb') as fp: short_context_p = pickle.load(fp)
            short_context_p = np.asarray(short_context_p)
            with open (FILE_LONG_CONTEXT_P, 'rb') as fp: long_context_p = pickle.load(fp)
            long_context_p = np.asarray(long_context_p)
            
            with open (FILE_SHORT_LASTITEM, 'rb') as fp: short_lastitem = pickle.load(fp)
            short_lastitem = np.asarray(short_lastitem)
            with open (FILE_SHORT_SEQLASTSESSION, 'rb') as fp: short_seqlastsession = pickle.load(fp)
            short_seqlastsession = np.asarray(short_seqlastsession)
            with open (FILE_LONG_LASTITEM, 'rb') as fp: long_lastitem = pickle.load(fp)
            long_lastitem = np.asarray(long_lastitem)
            
            with open (FILE_SHORT_CAT1, 'rb') as fp: short_X_cat1IDs = pickle.load(fp)
            short_X_cat1IDs = np.asarray(short_X_cat1IDs)
            with open (FILE_SHORT_CAT2, 'rb') as fp: short_X_cat2IDs = pickle.load(fp)
            short_X_cat2IDs = np.asarray(short_X_cat2IDs)
            with open (FILE_SHORT_CAT3, 'rb') as fp: short_X_cat3IDs = pickle.load(fp)
            short_X_cat3IDs = np.asarray(short_X_cat3IDs)
            with open (FILE_SHORT_CAT4, 'rb') as fp: short_X_cat4IDs = pickle.load(fp)
            short_X_cat4IDs = np.asarray(short_X_cat4IDs)
            with open (FILE_SHORT_CAT5, 'rb') as fp: short_X_cat5IDs = pickle.load(fp)
            short_X_cat5IDs = np.asarray(short_X_cat5IDs)
            with open (FILE_SHORT_CAT6, 'rb') as fp: short_X_cat6IDs = pickle.load(fp)
            short_X_cat6IDs = np.asarray(short_X_cat6IDs)
            with open (FILE_SHORT_CAT7, 'rb') as fp: short_X_cat7IDs = pickle.load(fp)
            short_X_cat7IDs = np.asarray(short_X_cat7IDs)
            with open (FILE_LONG_CAT1, 'rb') as fp: long_X_cat1IDs = pickle.load(fp)
            long_X_cat1IDs = np.asarray(long_X_cat1IDs)
            with open (FILE_LONG_CAT2, 'rb') as fp: long_X_cat2IDs = pickle.load(fp)
            long_X_cat2IDs = np.asarray(long_X_cat2IDs)
            
            with open (FILE_LONG_CAT3, 'rb') as fp: long_X_cat3IDs = pickle.load(fp)
            long_X_cat3IDs = np.asarray(long_X_cat3IDs)
            with open (FILE_LONG_CAT4, 'rb') as fp: long_X_cat4IDs = pickle.load(fp)
            long_X_cat4IDs = np.asarray(long_X_cat4IDs)
            with open (FILE_LONG_CAT5, 'rb') as fp: long_X_cat5IDs = pickle.load(fp)
            long_X_cat5IDs = np.asarray(long_X_cat5IDs)
            with open (FILE_LONG_CAT6, 'rb') as fp: long_X_cat6IDs = pickle.load(fp)
            long_X_cat6IDs = np.asarray(long_X_cat6IDs)
            with open (FILE_LONG_CAT7, 'rb') as fp: long_X_cat7IDs = pickle.load(fp)
            long_X_cat7IDs = np.asarray(long_X_cat7IDs)
            
            with open (FILE_OUTPUT, 'rb') as fp: y_IDs = pickle.load(fp)
            y_IDs = np.asarray(y_IDs)
            
            #--------------Output Tranform to 1-hot encoding for Softmax-----------#
           
            y_onehot = list()
            for value in y_IDs:
                value = int(value)-1     #index start from 0 in one-hot encoding
                letter = [0 for _ in range(TOTAL_ITEM)]
                letter[value] = 1
                y_onehot.append(letter)
            
            y_onehot = np.asarray(y_onehot, dtype=np.int32)
            
            
            """
            Model
            """
            #----------------- Attention Layer---------------------#
            class Attention_Short_Bottom(tf.keras.Model):
                def __init__(self, units, name):
                    super(Attention_Short_Bottom, self).__init__()
                    self.W1 = tf.keras.layers.Dense(units)  
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)       
                    self._name = name
                   
                def call(self, inputs):
                    # Step 1 - calculate attention scores
                    user_with_time_axis = inputs[:,-1:,:] 
                    features = inputs[:,:-1,:]
                    
                    score = tf.nn.tanh(self.W1(features) + self.W2(user_with_time_axis))
                    attention_weights = tf.nn.softmax(self.V(score), axis=1)
                    
                    #-------------Including Masking---------------#
                    # Step 2 - re-normalize the masked scores
                    masking_layer = Masking()                                   
                    masked_embedding = masking_layer(features)
                    mask = masked_embedding._keras_mask
                    mask = K.cast(mask, K.floatx())                    
                    mask = tf.expand_dims(mask, -1)                                       
                    masked_att = attention_weights * mask                
                    masked_att = tf.squeeze(masked_att,-1)                           
                    masked_att /= K.cast(K.sum(masked_att, axis=1, keepdims=True) + K.epsilon(), K.floatx())
                    a = K.expand_dims(masked_att)                                             
                    #--------------------------------------------#
                    # Step 3 - Weighted sum of hidden states, by the attention scores
                    context_vector = a * features
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, a
            
            class Attention_Short_Top(tf.keras.Model):
                def __init__(self, units, name):
                    super(Attention_Short_Top, self).__init__()
                    self.W1 = tf.keras.layers.Dense(units)  
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)       
                    self._name = name
                   
                def call(self, inputs):
                    # Step 1 - calculate attention scores
                    user = inputs[:,-2:-1,:] #user+lastitem
                    lastitem = inputs[:,-1:,:]
                    
                    user_with_time_axis = Concatenate(axis=2,name = 'short_context_inputs')([user, lastitem])
                    
                    features = inputs[:,:-2,:] 
                    
                    score = tf.nn.tanh(self.W1(features) + self.W2(user_with_time_axis))
                    attention_weights = tf.nn.softmax(self.V(score), axis=1)
                              #
                    #-------------Including Masking---------------#
                    # Step 2 - re-normalize the masked scores
                    masking_layer = Masking()                                   
                    masked_embedding = masking_layer(features)
                    mask = masked_embedding._keras_mask
                    mask = K.cast(mask, K.floatx())                                   
                    mask = tf.expand_dims(mask, -1)
                    masked_att = attention_weights * mask
                    masked_att = tf.squeeze(masked_att,-1)
                    masked_att /= K.cast(K.sum(masked_att, axis=1, keepdims=True) + K.epsilon(), K.floatx())
                    a = K.expand_dims(masked_att)
                    #--------------------------------------------#
                    # Step 3 - Weighted sum of hidden states, by the attention scores
                    context_vector = a * features
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, a
                
            class Attention_Long_Top(tf.keras.Model):
                def __init__(self, units, name):
                    super(Attention_Long_Top, self).__init__()
                    self.W1 = tf.keras.layers.Dense(units)  
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)       
                    self._name = name
                   
                def call(self, inputs):
                    #decompose inputs
                    user = inputs[:,-2:-1,:] #user+lastitem
                    lastitem = inputs[:,-1:,:]
                    user_with_time_axis = Concatenate(axis=2,name = 'long_context_inputs')([user, lastitem])
                    features = inputs[:,:-2,:] 

                    # Step 1 - calculate attention scores
                    score = tf.nn.tanh(self.W1(features) + self.W2(user_with_time_axis))
                    attention_weights = tf.nn.softmax(self.V(score), axis=1)
                    #-------------Including Masking---------------#
                    # Step 2 - re-normalize the masked scores
                    masking_layer = Masking()                                   
                    masked_embedding = masking_layer(features)    
                    mask = masked_embedding._keras_mask
                    mask = K.cast(mask, K.floatx())                                     
                    mask = tf.expand_dims(mask, -1)
                    masked_att = attention_weights * mask
                    masked_att = tf.squeeze(masked_att,-1)
                    masked_att /= K.cast(K.sum(masked_att, axis=1, keepdims=True) + K.epsilon(), K.floatx())
                    a = K.expand_dims(masked_att)
 
                    #--------------------------------------------#
                    # Step 3 - Weighted sum of hidden states, by the attention scores
                    context_vector = a * features
                    context_vector = tf.reduce_sum(context_vector, axis=1)  #(?,#sess,#features)
                    
                    context_vector = tf.expand_dims(context_vector, -1)     #(?,#sess,#features,1)
                    output = Concatenate(axis=1)([context_vector, a])       #(?,#sess,#features+#timestep,1)
                    return output
                
                def compute_output_shape(self, input_shape): #multiple outputs
                    """ Outputs produced by the layer """
                    return (input_shape[0], input_shape[2]+(input_shape[1]-2), 1) 
            
            class Attention_Long_Bottom(tf.keras.Model):
                def __init__(self, units, name):
                    super(Attention_Long_Bottom, self).__init__()
                    self.W1 = tf.keras.layers.Dense(units)  
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)       
                    self._name = name
                   
                def call(self, inputs):
                    #decompose inputs
                    features = inputs[:,:-1,:] 
                    user = inputs[:,-1:,:] #user
                    
                    
                    # Step 1 - calculate attention scores
                    
                    score = tf.nn.tanh(self.W1(features) + self.W2(user))
                    attention_weights = tf.nn.softmax(self.V(score), axis=1)
                    
                    #-------------Including Masking---------------#
                    # Step 2 - re-normalize the masked scores
                    masking_layer = Masking()                                   
                    masked_embedding = masking_layer(features)
                    mask = masked_embedding._keras_mask
                    mask = K.cast(mask, K.floatx())
                    mask = tf.expand_dims(mask, -1)
                    masked_att = attention_weights * mask
                    masked_att = tf.squeeze(masked_att,-1)
                    masked_att /= K.cast(K.sum(masked_att, axis=1, keepdims=True) + K.epsilon(), K.floatx())
                    a = K.expand_dims(masked_att)
                    #--------------------------------------------#
                    # Step 3 - Weighted sum of hidden states, by the attention scores
                    context_vector = a * features
                    context_vector = tf.reduce_sum(context_vector, axis=1)  #(?,#sess,#features)
                    
                    context_vector = tf.expand_dims(context_vector, -1)     #(?,#sess,#features,1)
                    output = Concatenate(axis=1)([context_vector, a])       #(?,#sess,#features+#timestep,1)

                    return output
                
                def compute_output_shape(self, input_shape): #multiple outputs
                    """ Outputs produced by the layer """
                    return (input_shape[0], input_shape[2]+(input_shape[1]-1), 1) #(?,#sess=2,#feature+#timestep7,1)

            class Attention_temporal(tf.keras.Model):
                def __init__(self, units):
                    super(Attention_temporal, self).__init__()
                    self.W1 = tf.keras.layers.Dense(units) 
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)
                    self._name = 'attention_temporal'
             
                def call(self, inputs):
                    long_session_vector = inputs[0]
                    u_short = inputs[1]
                    u_short_with_time_axis = tf.expand_dims(u_short, 1)
                    score = tf.nn.tanh(self.W1(long_session_vector) + self.W2(u_short_with_time_axis))
                    attention_weights = tf.nn.softmax(self.V(score), axis=1)
                    context_vector = attention_weights * long_session_vector
                    context_vector = tf.reduce_sum(context_vector, axis=1)
             
                    return context_vector, attention_weights
            """
            Create Model Graph
            """
            
            #----------------------Short_User----------------------#
            short_user_id = Input(name = 'short_user_id', shape = [1,])
            short_user_embedding = keras.layers.Embedding(name = 'short_user_embedding',
                                                          input_dim = TOTAL_USER,    
                                                          output_dim = INPUT_DIM,
                                                          input_length=1,
                                                          weights=[user_embedding],
                                                          trainable=False)(short_user_id)
           
            short_user_embedding = keras.layers.Flatten()(short_user_embedding) 

             #----------------------Short_LastItem----------------------#
            short_lastitem_id = Input(name = 'short_lastitem_id', shape = [1,])
            short_lastitem_embedding = keras.layers.Embedding(name = 'short_lastitem_embedding',
                                                          input_dim = TOTAL_ITEM+1,    
                                                          output_dim = INPUT_DIM,
                                                          input_length=1,
                                                          weights=[pretrained_weights_padded],
                                                          trainable=False)(short_lastitem_id)
        
            short_lastitem_embedding = keras.layers.Flatten()(short_lastitem_embedding) 

            #----------------------Long User----------------------#
            long_user_id = Input(name = 'long_user_id', shape = [N_session,1,])
            long_user_embedding = keras.layers.Embedding(input_dim = TOTAL_USER,     
                                                         output_dim = INPUT_DIM,
                                                         input_length=N_session,     
                                                         name = 'long_user_embedding',
                                                         weights=[user_embedding],
                                                         trainable=False)(long_user_id)
            
            #----------------------Long LastItem----------------------#
            long_lastitem_id = Input(name = 'long_lastitem_id', shape = [N_session,1,])
            long_lastitem_embedding = keras.layers.Embedding(input_dim = TOTAL_ITEM+1,     
                                                         output_dim = INPUT_DIM,
                                                         input_length=N_session,     
                                                         name = 'long_lastitem_embedding',
                                                         weights=[pretrained_weights_padded],
                                                         trainable=False)(long_lastitem_id)

            #--------------------Short Context ID------------------#
            short_sequence_context_id = Input(name = 'short_context_id', shape = [TOTAL_CONTEXT,], dtype='int32')  
            short_context_embedding = keras.layers.Embedding(TOTAL_CONTEXT,                
                                                       INPUT_DIM, 
                                                       input_length=TOTAL_CONTEXT,
                                                       weights=[context_embedding],
                                                       trainable=False,
                                                       name = 'short_contextid_embedding')(short_sequence_context_id)

            #---------------Short Context Percentage--------------#
            short_sequence_context_percent = Input(name = 'short_sequence_context_percent', shape = [TOTAL_CONTEXT,], dtype='int32') 
            short_context_percent_embedding = keras.layers.Embedding(TOTAL_PERCENT,       
                                                                     INPUT_DIM,           
                                                                     input_length=TOTAL_CONTEXT,
                                                                     weights=[context_percent_embedding],
                                                                     trainable=False,
                                                                     name = 'short_context_percent_embedding')(short_sequence_context_percent)

            #---------------Short Multiply Context----------------#
           
            short_context_multiply = keras.layers.Multiply(name = 'short_context_multiply')([short_context_embedding, short_context_percent_embedding])

            
            #--------------------Long Context ID------------------#
            long_sequence_context_id = Input(name = 'long_context_id',shape = [N_session,TOTAL_CONTEXT,], dtype='int32')  
            long_context_embedding = keras.layers.Embedding(TOTAL_CONTEXT,                
                                                            INPUT_DIM,                   
                                                            input_length=TOTAL_CONTEXT,
                                                            weights=[context_embedding],
                                                            trainable=False,
                                                            name = 'long_contextid_embedding')(long_sequence_context_id)
            

            #---------------Long Context Percentage--------------#
            long_sequence_context_percent = Input(name = 'long_sequence_context_percent', shape = [N_session,TOTAL_CONTEXT,], dtype='int32') 
            long_context_percent_embedding = keras.layers.Embedding(TOTAL_PERCENT,       
                                                                    INPUT_DIM,           
                                                                    input_length=TOTAL_CONTEXT,
                                                                    weights=[context_percent_embedding],
                                                                    trainable=False,
                                                                    name = 'long_context_percent_embedding')(long_sequence_context_percent)

            #---------------Long Multiply Context----------------#
            long_context_multiply = keras.layers.Multiply(name = 'long_context_multiply')([long_context_embedding, long_context_percent_embedding])
           
            #---------------Short Seq of Items on last session-------------------#
            short_sequence_items_lastsession_id = Input(name = 'short_sequence_items_lastsession_id', shape=(n,), dtype='int32')
            short_embedded_items_lastsession = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=n, 
                                                        trainable=False,
                                                        name='short_embedded_items_lastsession')(short_sequence_items_lastsession_id)
           
            #---------------Short Item Cat1-------------------#
            short_sequence_input_cat1 = Input(name = 'short_sequence_input_cat1', shape=(MAX_ITEM_CAT1,), dtype='int32')
            short_embedded_sequences_cat1 = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=MAX_ITEM_CAT1, 
                                                        trainable=False,
                                                        name='short_embedded_sequences_cat1')(short_sequence_input_cat1)

            
            #--------------Short Item Cat2------------------#
            short_sequence_input_cat2 = Input(name = 'short_sequence_input_cat2',shape=(MAX_ITEM_CAT2,), dtype='int32')
            short_embedded_sequences_cat2 = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=MAX_ITEM_CAT2, 
                                                        trainable=False,
                                                        name='short_embedded_sequences_cat2')(short_sequence_input_cat2)
          
            #---------------Short Item Cat3-------------------#
            short_sequence_input_cat3 = Input(name = 'short_sequence_input_cat3', shape=(MAX_ITEM_CAT3,), dtype='int32')
            short_embedded_sequences_cat3 = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=MAX_ITEM_CAT3, 
                                                        trainable=False,
                                                        name='short_embedded_sequences_cat3')(short_sequence_input_cat3)
           
            #--------------Short Item Cat4------------------#
            short_sequence_input_cat4 = Input(name = 'short_sequence_input_cat4',shape=(MAX_ITEM_CAT4,), dtype='int32')
            short_embedded_sequences_cat4 = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=MAX_ITEM_CAT4, 
                                                        trainable=False,
                                                        name='short_embedded_sequences_cat4')(short_sequence_input_cat4)
           
            #--------------Short Item Cat5------------------#
            short_sequence_input_cat5 = Input(name = 'short_sequence_input_cat5',shape=(MAX_ITEM_CAT5,), dtype='int32')
            short_embedded_sequences_cat5 = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=MAX_ITEM_CAT5, 
                                                        trainable=False,
                                                        name='short_embedded_sequences_cat5')(short_sequence_input_cat5)
            
            #--------------Short Item Cat6------------------#
            short_sequence_input_cat6 = Input(name = 'short_sequence_input_cat6',shape=(MAX_ITEM_CAT6,), dtype='int32')
            short_embedded_sequences_cat6 = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=MAX_ITEM_CAT6, 
                                                        trainable=False,
                                                        name='short_embedded_sequences_cat6')(short_sequence_input_cat6)
           #--------------Short Item Cat7------------------#
            short_sequence_input_cat7 = Input(name = 'short_sequence_input_cat7',shape=(MAX_ITEM_CAT7,), dtype='int32')
            short_embedded_sequences_cat7 = Embedding(TOTAL_ITEM+1, 
                                                        INPUT_DIM,   
                                                        weights=[pretrained_weights_padded],
                                                        input_length=MAX_ITEM_CAT7, 
                                                        trainable=False,
                                                        name='short_embedded_sequences_cat7')(short_sequence_input_cat7)
           
            
            #----------------Long Item Cat1--------------#
            long_sequence_input_cat1 = Input(name = 'long_sequence_input_cat1', shape=[N_session,MAX_ITEM_CAT1,], dtype='int32')
            long_embedded_sequences_cat1 = keras.layers.Embedding(TOTAL_ITEM+1,                         
                                                             INPUT_DIM,                             
                                                             weights=[pretrained_weights_padded],
                                                             input_length=MAX_ITEM_CAT1,   
                                                             trainable=False,
                                                             name='long_embedded_sequences_cat1')(long_sequence_input_cat1)
            
            
            #----------------Long Item Cat2--------------#
            long_sequence_input_cat2 = Input(name = 'long_sequence_input_cat2', shape=[N_session,MAX_ITEM_CAT2,], dtype='int32')
            long_embedded_sequences_cat2 = keras.layers.Embedding(TOTAL_ITEM+1,                         
                                                             INPUT_DIM,                             
                                                             weights=[pretrained_weights_padded],
                                                             input_length=MAX_ITEM_CAT2,   
                                                             trainable=False,
                                                             name='long_embedded_sequences_cat2')(long_sequence_input_cat2)

            
            #----------------Long Item Cat3--------------#
            long_sequence_input_cat3 = Input(name = 'long_sequence_input_cat3', shape=[N_session,MAX_ITEM_CAT3,], dtype='int32')
            long_embedded_sequences_cat3 = keras.layers.Embedding(TOTAL_ITEM+1,                         
                                                             INPUT_DIM,                             
                                                             weights=[pretrained_weights_padded],
                                                             input_length=MAX_ITEM_CAT3,   
                                                             trainable=False,
                                                             name='long_embedded_sequences_cat3')(long_sequence_input_cat3)

            
            #----------------Long Item Cat4--------------#
            long_sequence_input_cat4 = Input(name = 'long_sequence_input_cat4', shape=[N_session,MAX_ITEM_CAT4,], dtype='int32')
            long_embedded_sequences_cat4 = keras.layers.Embedding(TOTAL_ITEM+1,                         
                                                             INPUT_DIM,                            
                                                             weights=[pretrained_weights_padded],
                                                             input_length=MAX_ITEM_CAT4,   
                                                             trainable=False,
                                                             name='long_embedded_sequences_cat4')(long_sequence_input_cat4)

            
            #----------------Long Item Cat5--------------#
            long_sequence_input_cat5 = Input(name = 'long_sequence_input_cat5', shape=[N_session,MAX_ITEM_CAT5,], dtype='int32')
            long_embedded_sequences_cat5 = keras.layers.Embedding(TOTAL_ITEM+1,                         
                                                             INPUT_DIM,                             
                                                             weights=[pretrained_weights_padded],
                                                             input_length=MAX_ITEM_CAT5,   
                                                             trainable=False,
                                                             name='long_embedded_sequences_cat5')(long_sequence_input_cat5)

            
            #----------------Long Item Cat6--------------#
            long_sequence_input_cat6 = Input(name = 'long_sequence_input_cat6', shape=[N_session,MAX_ITEM_CAT6,], dtype='int32')
            long_embedded_sequences_cat6 = keras.layers.Embedding(TOTAL_ITEM+1,                         
                                                             INPUT_DIM,                             
                                                             weights=[pretrained_weights_padded],
                                                             input_length=MAX_ITEM_CAT6,   
                                                             trainable=False,
                                                             name='long_embedded_sequences_cat6')(long_sequence_input_cat6)

            
            #----------------Long Item Cat7--------------#
            long_sequence_input_cat7 = Input(name = 'long_sequence_input_cat7', shape=[N_session,MAX_ITEM_CAT7,], dtype='int32')
            long_embedded_sequences_cat7 = keras.layers.Embedding(TOTAL_ITEM+1,                         
                                                             INPUT_DIM,                             
                                                             weights=[pretrained_weights_padded],
                                                             input_length=MAX_ITEM_CAT7,   
                                                             trainable=False,
                                                             name='long_embedded_sequences_cat7')(long_sequence_input_cat7)

            
            #--------------Long Combine Inputs------------#
            long_inputs_context = Concatenate(axis=2,name = 'long_inputs_context')([long_context_multiply, long_user_embedding, long_lastitem_embedding])
            
            
            long_inputs_cat1 = Concatenate(axis=2,name = 'long_inputs_cat1')([long_embedded_sequences_cat1, long_user_embedding])
            
            
            long_inputs_cat2 = Concatenate(axis=2,name = 'long_inputs_cat2')([long_embedded_sequences_cat2, long_user_embedding])
            
            
            long_inputs_cat3 = Concatenate(axis=2,name = 'long_inputs_cat3')([long_embedded_sequences_cat3, long_user_embedding])
            
            
            long_inputs_cat4 = Concatenate(axis=2,name = 'long_inputs_cat4')([long_embedded_sequences_cat4, long_user_embedding])
           
            
            long_inputs_cat5 = Concatenate(axis=2,name = 'long_inputs_cat5')([long_embedded_sequences_cat5, long_user_embedding])
           
            
            long_inputs_cat6 = Concatenate(axis=2,name = 'long_inputs_cat6')([long_embedded_sequences_cat6, long_user_embedding])
           
            
            long_inputs_cat7 = Concatenate(axis=2,name = 'long_inputs_cat7')([long_embedded_sequences_cat7, long_user_embedding])
            
            
            """
            Attention Network for U_Short: u_short_embedding
            """
            
            #----------------Short Attention Context--------------#
            short_user_embedding = tf.expand_dims(short_user_embedding, 1)
            short_lastitem_embedding = tf.expand_dims(short_lastitem_embedding, 1)
            short_context_inputs = Concatenate(axis=1,name = 'short_context_inputs')([short_context_multiply, short_user_embedding, short_lastitem_embedding])
            
            short_attention_context = Attention_Short_Top(SHORT_ATTENTION_UNIT,'short_attention_context')
            short_context_vector_context, short_attention_weights_context = short_attention_context(short_context_inputs) 
            
            #-----------------Short Attention Cat1---------------#
            short_cat1_inputs = Concatenate(axis=1,name = 'short_cat1_inputs')([short_embedded_sequences_cat1, short_user_embedding])
            short_attention_cat1 = Attention_Short_Bottom(SHORT_ATTENTION_UNIT,'short_attention_cat1')
            short_context_vector_cat1, short_attention_weights_cat1 = short_attention_cat1(short_cat1_inputs) 
           
            #----------------Short Attention Cat2--------------#
            short_cat2_inputs = Concatenate(axis=1,name = 'short_cat2_inputs')([short_embedded_sequences_cat2, short_user_embedding])
            short_attention_cat2 = Attention_Short_Bottom(SHORT_ATTENTION_UNIT,'short_attention_cat2')
            short_context_vector_cat2, short_attention_weights_cat2  = short_attention_cat2(short_cat2_inputs) 
           
            #----------------Short Attention Cat3--------------#
            short_cat3_inputs = Concatenate(axis=1,name = 'short_cat3_inputs')([short_embedded_sequences_cat3, short_user_embedding])
            short_attention_cat3 = Attention_Short_Bottom(SHORT_ATTENTION_UNIT,'short_attention_cat3')
            short_context_vector_cat3, short_attention_weights_cat3  = short_attention_cat3(short_cat3_inputs) 
            
            #----------------Short Attention Cat4--------------#
            short_cat4_inputs = Concatenate(axis=1,name = 'short_cat4_inputs')([short_embedded_sequences_cat4, short_user_embedding])
            short_attention_cat4 = Attention_Short_Bottom(SHORT_ATTENTION_UNIT,'short_attention_cat4')
            short_context_vector_cat4, short_attention_weights_cat4  = short_attention_cat4(short_cat4_inputs) 
            
            #----------------Short Attention Cat5--------------#
            short_cat5_inputs = Concatenate(axis=1,name = 'short_cat5_inputs')([short_embedded_sequences_cat5, short_user_embedding])
            short_attention_cat5 = Attention_Short_Bottom(SHORT_ATTENTION_UNIT,'short_attention_cat5')
            short_context_vector_cat5, short_attention_weights_cat5  = short_attention_cat5(short_cat5_inputs) 
           
            #----------------Short Attention Cat6--------------#
            short_cat6_inputs = Concatenate(axis=1,name = 'short_cat6_inputs')([short_embedded_sequences_cat6, short_user_embedding])
            short_attention_cat6 = Attention_Short_Bottom(SHORT_ATTENTION_UNIT,'short_attention_cat6')
            short_context_vector_cat6, short_attention_weights_cat6  = short_attention_cat6(short_cat6_inputs) 
           
            #----------------Short Attention Cat7--------------#
            short_cat7_inputs = Concatenate(axis=1,name = 'short_cat7_inputs')([short_embedded_sequences_cat7, short_user_embedding])
            short_attention_cat7 = Attention_Short_Bottom(SHORT_ATTENTION_UNIT,'short_attention_cat7')
            short_context_vector_cat7, short_attention_weights_cat7  = short_attention_cat7(short_cat7_inputs) 
           
            #----------------Short : Merge all item latent in different relations--------------#
            
            def short_calculate_sessionvector(x):
                attention_weights_1statt = x[0]
                item_latent_p = Concatenate(axis=1)([K.expand_dims(x[1], 1),K.expand_dims(x[2], 1),K.expand_dims(x[3], 1),K.expand_dims(x[4], 1),K.expand_dims(x[5], 1),K.expand_dims(x[6], 1),K.expand_dims(x[7], 1)])  # [B,#cat,H]
                context_vector_mul = attention_weights_1statt * item_latent_p
                context_vector = tf.reduce_sum(context_vector_mul, axis=1)
                return context_vector, item_latent_p, attention_weights_1statt
            
            short_session_vector, short_item_latent_p, short_attention_weights_1statt = Lambda(short_calculate_sessionvector,name = 'short_session_vector')([short_attention_weights_context,
                                                                             short_context_vector_cat3,short_context_vector_cat4,
             
            u_short_embedding = short_session_vector
            
            """
            GRU Network for short Sequential model: u_seq
            """
            gru = tf.keras.layers.GRU(RNN_CELL_UNIT, return_sequences=True, return_state=True, name="u_seq_gru")
            u_seq_whole_sequence_output, u_seq_final_state = gru(short_embedded_items_lastsession)
            
            """
            Attention Network for U_Long: u_long_embedding
            """
            
            #--------------Long Attention Class context-----------#
            long_attention_context = Attention_Long_Top(units=LONG_ATTENTION_UNIT,name='long_attention_context')
            long_outputs_context = TimeDistributed(long_attention_context)(long_inputs_context)
            long_session_att_context = long_outputs_context[:,:,INPUT_DIM:,:]                 #shape=(None, 2, 2, 1)
         
            #--------------Long Attention Class Cat1-----------#
            long_attention_cat1 = Attention_Long_Bottom(units=LONG_ATTENTION_UNIT,name='long_attention_cat1')
            long_outputs_cat1 = TimeDistributed(long_attention_cat1)(long_inputs_cat1)
            long_session_context_cat1 = long_outputs_cat1[:,:,:INPUT_DIM,:]                   #shape=(None, 2, 3, 1)
            long_session_context_cat1 = tf.squeeze(long_session_context_cat1, axis=-1)
            long_session_att_cat1 = long_outputs_cat1[:,:,INPUT_DIM:,:]                       #shape=(None, 2, 4, 1)

            #--------------Long Attention Class Cat2-----------#
            long_attention_cat2 = Attention_Long_Bottom(units=LONG_ATTENTION_UNIT,name='long_attention_cat2')
            long_outputs_cat2 = TimeDistributed(long_attention_cat2)(long_inputs_cat2)
            long_session_context_cat2 = long_outputs_cat2[:,:,:INPUT_DIM,:]                   #shape=(None, 2, 3, 1)
            long_session_context_cat2 = tf.squeeze(long_session_context_cat2, axis=-1)
            long_session_att_cat2 = long_outputs_cat2[:,:,INPUT_DIM:,:]                       #shape=(None, 2, 4, 1)

            #--------------Long Attention Class Cat3-----------#
            long_attention_cat3 = Attention_Long_Bottom(units=LONG_ATTENTION_UNIT,name='long_attention_cat3')
            long_outputs_cat3 = TimeDistributed(long_attention_cat3)(long_inputs_cat3)
            long_session_context_cat3 = long_outputs_cat3[:,:,:INPUT_DIM,:]                   #shape=(None, 2, 3, 1)
            long_session_context_cat3 = tf.squeeze(long_session_context_cat3, axis=-1)
            long_session_att_cat3 = long_outputs_cat3[:,:,INPUT_DIM:,:]                       #shape=(None, 2, 4, 1)
            
            #--------------Long Attention Class Cat4-----------#
            long_attention_cat4 = Attention_Long_Bottom(units=LONG_ATTENTION_UNIT,name='long_attention_cat4')
            long_outputs_cat4 = TimeDistributed(long_attention_cat4)(long_inputs_cat4)
            long_session_context_cat4 = long_outputs_cat4[:,:,:INPUT_DIM,:]                   #shape=(None, 2, 3, 1)
            long_session_context_cat4 = tf.squeeze(long_session_context_cat4, axis=-1)
            long_session_att_cat4 = long_outputs_cat4[:,:,INPUT_DIM:,:]                       #shape=(None, 2, 4, 1)

            #--------------Long Attention Class Cat5-----------#
            long_attention_cat5 = Attention_Long_Bottom(units=LONG_ATTENTION_UNIT,name='long_attention_cat5')
            long_outputs_cat5 = TimeDistributed(long_attention_cat5)(long_inputs_cat5)
            long_session_context_cat5 = long_outputs_cat5[:,:,:INPUT_DIM,:]                   #shape=(None, 2, 3, 1)
            long_session_context_cat5 = tf.squeeze(long_session_context_cat5, axis=-1)
            long_session_att_cat5 = long_outputs_cat5[:,:,INPUT_DIM:,:]                       #shape=(None, 2, 4, 1)

            #--------------Long Attention Class Cat6-----------#
            long_attention_cat6 = Attention_Long_Bottom(units=LONG_ATTENTION_UNIT,name='long_attention_cat6')
            long_outputs_cat6 = TimeDistributed(long_attention_cat6)(long_inputs_cat6)
            long_session_context_cat6 = long_outputs_cat6[:,:,:INPUT_DIM,:]                   #shape=(None, 2, 3, 1)
            long_session_context_cat6 = tf.squeeze(long_session_context_cat6, axis=-1)
            long_session_att_cat6 = long_outputs_cat6[:,:,INPUT_DIM:,:]                       #shape=(None, 2, 4, 1)

            #--------------Long Attention Class Cat7-----------#
            long_attention_cat7 = Attention_Long_Bottom(units=LONG_ATTENTION_UNIT,name='long_attention_cat7')
            long_outputs_cat7 = TimeDistributed(long_attention_cat7)(long_inputs_cat7)
            long_session_context_cat7 = long_outputs_cat7[:,:,:INPUT_DIM,:]                   #shape=(None, 2, 3, 1)
            long_session_context_cat7 = tf.squeeze(long_session_context_cat7, axis=-1)
            long_session_att_cat7 = long_outputs_cat7[:,:,INPUT_DIM:,:]                       #shape=(None, 2, 4, 1)

            #--------(Long-term part) Merge all item latent in different relations--------------#
           
            def long_calculate_sessionvector(x):
                long_attention_weights = x[0]
                long_item_latent_p = Concatenate(axis=2)([K.expand_dims(x[1], 2),K.expand_dims(x[2], 2),K.expand_dims(x[3], 2),
                                                            K.expand_dims(x[4], 2),K.expand_dims(x[5], 2),K.expand_dims(x[6], 2),K.expand_dims(x[7], 2)])  #[B,#sess,#cat,#feature]
                long_context_vector_mul = long_attention_weights * long_item_latent_p                      #[B,#sess,#cat,#feature]
                long_context_vector = tf.reduce_sum(long_context_vector_mul, axis=2)                       #[B,#sess,#feature]
                return long_context_vector,long_attention_weights,x[1],x[2]
            
            long_session_vector,long_attention_weights,session_context_longcat1,session_context_longcat2  = Lambda(long_calculate_sessionvector,name = 'longsession_vector')([long_session_att_context,long_session_context_cat1,long_session_context_cat2,long_session_context_cat3,long_session_context_cat4,long_session_context_cat5,long_session_context_cat6,long_session_context_cat7])

            """
            BI-GRU for U_Long: output attention weights for each timestep & u_long_embedding
            """
            #------Bi-directional LSTM-------#
            gru, forward_h, backward_h = tf.keras.layers.Bidirectional(tf.keras.layers.GRU
                                                 (RNN_CELL_UNIT,
                                                  #dropout=RNN_CELL_DROPOUT,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_activation=RNN_CELL_ACTIVATION,
                                                  recurrent_initializer=RNN_CELL_INITIALIZER), 
                                                  name="bi_gru")(long_session_vector)

            attention_temporal = Attention_temporal(TEMPORAL_ATTENTION_UNIT)
            temporal_context_vector, temporal_attention_weights = attention_temporal([gru, u_short_embedding])

            u_long_embedding = temporal_context_vector
            
            """
            Concat u_long_embedding & u_short_embedding into user_vector & u_seq_last_session--> Dense
            """
            user_concatvector = Concatenate()([u_long_embedding, u_short_embedding, u_seq_final_state])
            user_vector = keras.layers.Dense(FINAL_DENSE_UNIT, activation=FINAL_DENSE_ACTIVATION)(user_concatvector)
 
            #---------------Conclude Model----------------#
            
           
            output = keras.layers.Dense(TOTAL_ITEM, activation='softmax')(user_vector)
            model = keras.Model(inputs=[short_user_id, short_lastitem_id, short_sequence_items_lastsession_id ,short_sequence_input_cat1,short_sequence_input_cat2,
                                        short_sequence_input_cat3,short_sequence_input_cat4,short_sequence_input_cat5,short_sequence_input_cat6,short_sequence_input_cat7,
                                        short_sequence_context_id,short_sequence_context_percent,
                                        long_user_id,long_lastitem_id, long_sequence_input_cat1,long_sequence_input_cat2,
                                        long_sequence_input_cat3,long_sequence_input_cat4,long_sequence_input_cat5,long_sequence_input_cat6,long_sequence_input_cat7,
                                        long_sequence_context_id,long_sequence_context_percent], 
                                        outputs=output)

            """
            Compile Model Graph
            """
            #----------compile model-------#
            model.compile(optimizer=tf.optimizers.Adam(),  #tf.train.AdamOptimizer() for tf1.1
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    min_delta=0,
                                                                    patience=10,
                                                                    verbose=0, mode='auto')
            """
            Train Model Graph
            """
            #----------train model--------#
            
            history = model.fit([short_user_input, short_lastitem, short_seqlastsession, short_X_cat1IDs,short_X_cat2IDs,short_X_cat3IDs,short_X_cat4IDs,short_X_cat5IDs,short_X_cat6IDs,short_X_cat7IDs,
                                 short_context_input,short_context_p,
                                 long_user_input,long_lastitem, long_X_cat1IDs,long_X_cat2IDs,long_X_cat3IDs,long_X_cat4IDs,long_X_cat5IDs,long_X_cat6IDs,long_X_cat7IDs,long_context_input,long_context_p],
                                y_onehot,
                                epochs=MODEL_TRAIN_EPOCH, #MODEL_EPOCH 5
                                batch_size=MODEL_BATCH, #MODEL_BATCH 1
                                validation_split=.1, verbose=1, callbacks=[early_stopping_callback])
            
            
            with open(FILE_MODEL_LOSS, 'w') as file: file.write(str(history.history['loss']))
            with open(FILE_MODEL_ACC, 'w') as file: file.write(str(history.history['accuracy']))
            with open(FILE_MODEL_VALLOSS, 'w') as file: file.write(str(history.history['val_loss']))
            with open(FILE_MODEL_VALACC, 'w') as file: file.write(str(history.history['val_accuracy']))
            
            #create loss plot
            pyplot.plot(history.history['loss'])
            pyplot.plot(history.history['val_loss'])
            pyplot.title('model train vs validation loss')
            pyplot.ylabel('loss')
            pyplot.xlabel('epoch')
            pyplot.legend(['train','validation'], loc= 'upper right')
            #pyplot.show()
            pyplot.savefig(FILE_MODEL_LOSSPNG)
            pyplot.clf()
            """
            Save Model Graph
            https://www.tensorflow.org/tutorials/keras/save_and_load
            """
            print("--------------------Save Model Graph---------------------")
            tf.keras.models.save_model(
                model,
                FILE_MODEL_SAVE,
                overwrite=True,
                include_optimizer=True
            )
            
            """
            Test Set
            """
           
            #----------Files for Test-----------#
            FILE_TEST_ROOT = 'datasets/gw_test/'
            FILE_SHORT_USER = FILE_TEST_ROOT+'short_user.txt'
            FILE_SHORT_CONTEXT = FILE_TEST_ROOT+'short_context.txt'
            FILE_SHORT_CONTEXT_P = FILE_TEST_ROOT+'short_context_p.txt'
            FILE_SHORT_LASTITEM = FILE_TEST_ROOT+'lastitem.txt'
            FILE_SHORT_SEQLASTSESSION = FILE_TEST_ROOT+'seqlastsession.txt'
            FILE_SHORT_CAT1 = FILE_TEST_ROOT+'short_itemcat1.txt'
            FILE_SHORT_CAT2 = FILE_TEST_ROOT+'short_itemcat2.txt'
            FILE_SHORT_CAT3 = FILE_TEST_ROOT+'short_itemcat3.txt'
            FILE_SHORT_CAT4 = FILE_TEST_ROOT+'short_itemcat4.txt'
            FILE_SHORT_CAT5 = FILE_TEST_ROOT+'short_itemcat5.txt'
            FILE_SHORT_CAT6 = FILE_TEST_ROOT+'short_itemcat6.txt'
            FILE_SHORT_CAT7 = FILE_TEST_ROOT+'short_itemcat7.txt'
            FILE_LONG_USER = FILE_TEST_ROOT+'long_user.txt'
            FILE_LONG_LASTITEM = FILE_TEST_ROOT+'long_lastitem.txt'
            FILE_LONG_CONTEXT = FILE_TEST_ROOT+'long_context.txt'
            FILE_LONG_CONTEXT_P = FILE_TEST_ROOT+'long_context_p.txt'
            FILE_LONG_CAT1 = FILE_TEST_ROOT+'long_itemcat1.txt'
            FILE_LONG_CAT2 = FILE_TEST_ROOT+'long_itemcat2.txt'
            FILE_LONG_CAT3 = FILE_TEST_ROOT+'long_itemcat3.txt'
            FILE_LONG_CAT4 = FILE_TEST_ROOT+'long_itemcat4.txt'
            FILE_LONG_CAT5 = FILE_TEST_ROOT+'long_itemcat5.txt'
            FILE_LONG_CAT6 = FILE_TEST_ROOT+'long_itemcat6.txt'
            FILE_LONG_CAT7 = FILE_TEST_ROOT+'long_itemcat7.txt'
            FILE_OUTPUT = FILE_TEST_ROOT+'itemoutput.txt'
            
            #----------Files for Output from Testing-----------#
            FILE_TESTOUTPUT = 'datasets/gw_test/result.txt'
            FILE_MODEL_TEST_SAVE = 'datasets/gw_test/gw_CReS'
            
            """
            Prepare Input & Output Data for Testing
            """
            #-----------Load All Variables-----------#
            #short_user_input= []
            with open (FILE_SHORT_USER, 'rb') as fp: short_user_input = pickle.load(fp)
            short_user_input = np.asarray(short_user_input)
            #print("short_user_input: ",short_user_input)
            with open (FILE_LONG_USER, 'rb') as fp: long_user_input = pickle.load(fp)
            long_user_input = np.asarray(long_user_input)
            with open (FILE_SHORT_CONTEXT, 'rb') as fp: short_context_input = pickle.load(fp)
            short_context_input = np.asarray(short_context_input)
            with open (FILE_LONG_CONTEXT, 'rb') as fp: long_context_input = pickle.load(fp)
            long_context_input = np.asarray(long_context_input)
            with open (FILE_SHORT_CONTEXT_P, 'rb') as fp: short_context_p = pickle.load(fp)
            short_context_p = np.asarray(short_context_p)
            with open (FILE_LONG_CONTEXT_P, 'rb') as fp: long_context_p = pickle.load(fp)
            long_context_p = np.asarray(long_context_p)
            
            with open (FILE_SHORT_LASTITEM, 'rb') as fp: short_lastitem = pickle.load(fp)
            short_lastitem = np.asarray(short_lastitem)
            with open (FILE_SHORT_SEQLASTSESSION, 'rb') as fp: short_seqlastsession = pickle.load(fp)
            short_seqlastsession = np.asarray(short_seqlastsession)
            with open (FILE_LONG_LASTITEM, 'rb') as fp: long_lastitem = pickle.load(fp)
            long_lastitem = np.asarray(long_lastitem)
            
            with open (FILE_SHORT_CAT1, 'rb') as fp: short_X_cat1IDs = pickle.load(fp)
            short_X_cat1IDs = np.asarray(short_X_cat1IDs)
            with open (FILE_SHORT_CAT2, 'rb') as fp: short_X_cat2IDs = pickle.load(fp)
            short_X_cat2IDs = np.asarray(short_X_cat2IDs)
            with open (FILE_SHORT_CAT3, 'rb') as fp: short_X_cat3IDs = pickle.load(fp)
            short_X_cat3IDs = np.asarray(short_X_cat3IDs)
            with open (FILE_SHORT_CAT4, 'rb') as fp: short_X_cat4IDs = pickle.load(fp)
            short_X_cat4IDs = np.asarray(short_X_cat4IDs)
            with open (FILE_SHORT_CAT5, 'rb') as fp: short_X_cat5IDs = pickle.load(fp)
            short_X_cat5IDs = np.asarray(short_X_cat5IDs)
            with open (FILE_SHORT_CAT6, 'rb') as fp: short_X_cat6IDs = pickle.load(fp)
            short_X_cat6IDs = np.asarray(short_X_cat6IDs)
            with open (FILE_SHORT_CAT7, 'rb') as fp: short_X_cat7IDs = pickle.load(fp)
            short_X_cat7IDs = np.asarray(short_X_cat7IDs)
            with open (FILE_LONG_CAT1, 'rb') as fp: long_X_cat1IDs = pickle.load(fp)
            long_X_cat1IDs = np.asarray(long_X_cat1IDs)
            with open (FILE_LONG_CAT2, 'rb') as fp: long_X_cat2IDs = pickle.load(fp)
            long_X_cat2IDs = np.asarray(long_X_cat2IDs)
            with open (FILE_LONG_CAT3, 'rb') as fp: long_X_cat3IDs = pickle.load(fp)
            long_X_cat3IDs = np.asarray(long_X_cat3IDs)
            with open (FILE_LONG_CAT4, 'rb') as fp: long_X_cat4IDs = pickle.load(fp)
            long_X_cat4IDs = np.asarray(long_X_cat4IDs)
            with open (FILE_LONG_CAT5, 'rb') as fp: long_X_cat5IDs = pickle.load(fp)
            long_X_cat5IDs = np.asarray(long_X_cat5IDs)
            with open (FILE_LONG_CAT6, 'rb') as fp: long_X_cat6IDs = pickle.load(fp)
            long_X_cat6IDs = np.asarray(long_X_cat6IDs)
            with open (FILE_LONG_CAT7, 'rb') as fp: long_X_cat7IDs = pickle.load(fp)
            long_X_cat7IDs = np.asarray(long_X_cat7IDs)
            with open (FILE_OUTPUT, 'rb') as fp: y_IDs = pickle.load(fp)
            y_IDs = np.asarray(y_IDs)
            #--------------Output Tranform to 1-hot encoding for Softmax-----------#
            #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
            y_onehot = list()
            for value in y_IDs:
                value = int(value)-1     #index start from 0 in one-hot encoding
                letter = [0 for _ in range(TOTAL_ITEM)]
                letter[value] = 1
                y_onehot.append(letter)
            
            y_onehot = np.asarray(y_onehot, dtype=np.int32)
            
            
            
            """
            Prediction Model (topN Recommendation) from SoftMax: H@N, MRR@N
            """
            def mean_reciprocal_rank(rs,r):
                try:
                    index = list(rs).index(r) 
                    return 1/float(index+1)
                except:
                    return 0.0
                
            def hit_rate(result_list, ground_truth):
                hit = 0
                total_sample = len(result_list)
                for i in range(len(result_list)):
                    if ground_truth[i] in result_list[i]:
                        hit +=1
                return hit/total_sample
                
            result = model.predict([short_user_input,short_lastitem, short_seqlastsession, short_X_cat1IDs,short_X_cat2IDs,short_X_cat3IDs,short_X_cat4IDs,short_X_cat5IDs,short_X_cat6IDs,short_X_cat7IDs,
                                 short_context_input,short_context_p,
                                 long_user_input,long_lastitem,long_X_cat1IDs,long_X_cat2IDs,long_X_cat3IDs,long_X_cat4IDs,long_X_cat5IDs,
                                 long_X_cat6IDs,long_X_cat7IDs,long_context_input,long_context_p])
            
            
            result = np.array(result)
            result_list = []
            ground_truth = []
            mrr_list = []       #hitrarte calcaulated from mrr !=0
            
            for i in range(len(result)):
                result_list.append(result[i].argsort()[-N:][::-1]+1)  #topN of index
                ground_truth.append(argmax(y_onehot[i, :])+1)         #groundtruth label of index
                mrr_list.append(mean_reciprocal_rank(result[i].argsort()[-N:][::-1],argmax(y_onehot[i, :])))
            
           
            h_n = hit_rate(result_list, ground_truth)
            mrr_n = mean(mrr_list)

            with open(FILE_TESTOUTPUT, "w") as text_file:
                    print('h_n: {}\nmrr_n: {}\n'.
                          format(h_n,mrr_n), file=text_file)
