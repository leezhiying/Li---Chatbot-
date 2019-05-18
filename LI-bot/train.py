#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:04:07 2019

@author: lizhiying
"""

import numpy as np 
import matplotlib.pyplot as plt
import math 
import nltk 
import tensorflow as tf 
import os 
import io 
import re 
import time 
import unicodedata
from collections import Counter
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.chdir('/Users/lizhiying/Desktop/Big Data Analytics/LI-bot')

from sklearn.model_selection import train_test_split
import keras 
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import random
random.seed(0)

##################################
print("loading data")
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
MAX_VOCAB_SIZE = 2000
NUM_SAMPLE = 10000
P_UNK = 0.5

def load_data(input_path,target_path):

    lines_input = open(input_path, 'rt', encoding='utf8').read().split('\n')
    lines_target = open(target_path, 'rt', encoding='utf8').read().split('\n')
    input_texts = []
    target_texts = []
    prev_words = []
    
    input_counter = Counter()
    target_counter = Counter()
    

    
    for line in lines_input:
    
        next_words = [w.lower() for w in nltk.word_tokenize(line)]
        input_texts.append(next_words)
        #if len(next_words) > MAX_TARGET_SEQ_LENGTH:
            #next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]
     
        for w in prev_words:
            input_counter[w] += 1
        prev_words = next_words
     
    
    prev_words = []
     
    for line in lines_target:
        next_words = [w.lower() for w in nltk.word_tokenize(line)]
        target_words = next_words
        target_words.insert(0, 'START')
        target_words.append('END')
        target_texts.append(next_words)
        
        for w in prev_words:
            target_counter[w] += 1
        prev_words = next_words
        
    print(input_texts[1000],target_texts[1000])
    
    input_texts_new = []
    target_texts_new = []
    for i in range(len(input_texts)-1):
        if len(input_texts[i]) < MAX_INPUT_SEQ_LENGTH and len(target_texts[i]) < MAX_TARGET_SEQ_LENGTH and len(input_texts[i])>1 and len(target_texts[i])>1:       
            input_texts_new.append(input_texts[i])
            target_texts_new.append(target_texts[i])

    return input_texts_new, target_texts_new, input_counter, target_counter

cornell_input , cornell_output, cornell_input_counter, cornell_output_counter = load_data("data/cornell.in","data/cornell.ou")
squad_input, squad_output, squad_input_counter, squad_output_counter = load_data("data/squad.in","data/squad.ou")
wiki_input, wiki_output, wiki_input_counter, wiki_counter = load_data("data/wiki.in","data/wiki.ou")




input_texts_new = cornell_input + squad_input + wiki_input 
target_texts_new = cornell_output + squad_output + wiki_output

input_counter = cornell_input_counter + squad_input_counter + wiki_input_counter
target_counter  = cornell_output_counter + squad_output_counter + wiki_output_counter

#####Tuning parameter#######
  
random.seed(0)
#input_texts_new = random.sample(input_texts_new,NUM_SAMPLE)
#target_texts_new = random.sample(target_texts_new,NUM_SAMPLE)
        
        
#input_texts_new,_, target_texts_new,_ = train_test_split(input_texts_new, target_texts_new, test_size=0.9, random_state=42)



print(input_texts_new[1])
print(target_texts_new[1])




input_word2idx = dict()
target_word2idx = dict()
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_word2idx[word[0]] = idx + 2
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 1

input_word2idx['PAD'] = 0
input_word2idx['UNK'] = 1
target_word2idx['UNK'] = 0

input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

num_encoder_tokens = len(input_idx2word)
num_decoder_tokens = len(target_idx2word)

np.save('models/cornell/word-input-word2idx.npy', input_word2idx)
np.save('models/cornell/word-input-idx2word.npy', input_idx2word)
np.save('models/cornell/word-target-word2idx.npy', target_word2idx)
np.save('models/cornell/word-target-idx2word.npy', target_idx2word)

encoder_input_data = []
 



encoder_max_seq_length = 0
decoder_max_seq_length = 0



def rand_pick(seq , probabilities):
    x = random.uniform(0 ,1)
    cumprob = 0.0
    for item , item_pro in zip(seq , probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item
 








for input_words, target_words in zip(input_texts_new, target_texts_new):
    encoder_input_wids = []
    for w in input_words:
        w2idx = 1  # default [UNK]
        if w in input_word2idx:
            w2idx = input_word2idx[w]
        if w2idx == 1:
            q = rand_pick([1,0],[P_UNK,1-P_UNK])
            if q == 1:
                encoder_input_wids.append(w2idx)
        else:
            encoder_input_wids.append(w2idx)

    encoder_input_data.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

print(context)
np.save('models/cornell/word-context.npy', context)


a = 0
for item in encoder_input_data:
    if 1 in item:
        a += 1

print(a)
        

target_new = []
for i in range(len(target_texts_new)):
    l = []
    item = target_texts_new[i]
    for j in range(len(item)):
        if item[j] in target_word2idx:
            l.append(item[j])
        else:
            q = rand_pick([1,0],[P_UNK,1-P_UNK])
            if q == 1:
                l.append(item[j])
    target_new.append(l)
            

    
target_texts_new = target_new 













##########






 
print("loading data complete. Now create model!")



#############################################
#### Train the model and tuning the parameters 
BATCH_SIZE = 64
NUM_EPOCHS = 1
HIDDEN_UNITS = 256
#optimizer = 'adam'
#WEIGHT_FILE_PATH = 'models/cornell/word-weights.h5'
WEIGHT_FILE_PATH = 'parameters/word-weights.h5'
optimizer = 'rmsprop'



def train(epochs=NUM_EPOCHS, BATCH_SIZE = BATCH_SIZE, optimizer = optimizer):
    start = time.time()
    
    def generate_batch(input_data, output_text_data):
        num_batches = len(input_data) // BATCH_SIZE
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * BATCH_SIZE
                end = (batchIdx + 1) * BATCH_SIZE
                encoder_input_data_batch = pad_sequences(input_data[start:end], encoder_max_seq_length)
                decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
                decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
                for lineIdx, target_words in enumerate(output_text_data[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in target_word2idx:
                            w2idx = target_word2idx[w]
                        decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                        if idx > 0:
                            decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1

                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch
    
    
    
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=HIDDEN_UNITS,
                                  input_length=encoder_max_seq_length, name='encoder_embedding')
    encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
    #encoder_lstm = GRU(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
    encoder_states = [encoder_state_h, encoder_state_c]
    
    
    
    
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
    decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
    #decoder_lstm = GRU(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
    
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                     initial_state=encoder_states)
    decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    
    
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    print(model.summary())
    
    
    json = model.to_json()
    open('models/cornell/word-architecture.json', 'w').write(json)
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(encoder_input_data, target_texts_new, test_size=0.2, random_state=42)
    
    print(len(Xtrain))
    print(len(Xtest))
    
    train_gen = generate_batch(Xtrain, Ytrain)
    test_gen = generate_batch(Xtest, Ytest)
    
    train_num_batches = len(Xtrain) // BATCH_SIZE
    test_num_batches = len(Xtest) // BATCH_SIZE
    
    checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
    h = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                        epochs=NUM_EPOCHS,
                        verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])
    
    model.save_weights(WEIGHT_FILE_PATH)
    end = time.time()
    
    run_time = end - start 
    
    print(run_time)
    return h.history['val_loss'], run_time
    

'''
BATCH_SIZE_list = [20,30,50,100]
model_val_loss = []
model_run_time = []
for i in range(len(BATCH_SIZE_list)):
    print(BATCH_SIZE_list[i])
    batch_size = BATCH_SIZE_list[i]
    a,b = train(NUM_EPOCHS,BATCH_SIZE= batch_size,optimizer = 'adam')
    model_val_loss.append(a)
    model_run_time.append(b)

plt.plot(BATCH_SIZE_list,model_val_loss)
'''   
BATCH_SIZE =50
train(NUM_EPOCHS,BATCH_SIZE,optimizer)














 




 




