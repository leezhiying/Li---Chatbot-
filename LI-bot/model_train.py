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
os.chdir('/Users/lizhiying/Desktop/Big Data Analytics/LI-master')

from sklearn.model_selection import train_test_split
import keras 
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences
import pandas as pd 
from keras.utils import to_categorical
 

class Chatbot: 
    def __init__(self):
        self.data = [] 


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

num_examples = None

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    
    word_pairs = [x[0] for x in word_pairs]
    
    #t = tuple(word_pairs)
    
    return word_pairs 






def tokenize(lang):
  lang_tokenizer = keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)
  
  tensor = lang_tokenizer.texts_to_sequences(lang)
  
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                      padding='post')
  #one_hot_results = lang_tokenizer.texts_to_matrix(lang,mode = "binary")
  
  return tensor, lang_tokenizer#, one_hot_results


##################################
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
MAX_VOCAB_SIZE = 800

#DATA_PATH = "data/data.in"
lines_input = open("data/data.in", 'rt', encoding='utf8').read().split('\n')
lines_target = open("data/data.ou", 'rt', encoding='utf8').read().split('\n')
input_texts = []
target_texts = []
prev_words = []

input_counter = Counter()
target_counter = Counter()




for line in lines_input:

    next_words = [w.lower() for w in nltk.word_tokenize(line)]
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

    if len(prev_words) > 0:
        input_texts.append(prev_words)
        for w in prev_words:
            input_counter[w] += 1

        #target_words = next_words[:]
        #target_words.insert(0, 'START')
        #arget_words.append('END')
        #for w in target_words:
            #target_counter[w] += 1
        #target_texts.append(target_words)

    prev_words = next_words


prev_words = []

for line in lines_target:
    next_words = [w.lower() for w in nltk.word_tokenize(line)]
        
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

    if len(prev_words) > 0:
        prev_words.insert(0,'START')
        prev_words.append('END')
        target_texts.append(prev_words)
        for w in prev_words:
            target_counter[w] += 1

        #target_words = next_words[:]
        #target_words.insert(0, 'START')
        #arget_words.append('END')
        #for w in target_words:
            #target_counter[w] += 1
        #target_texts.append(target_words)

    prev_words = next_words
    
    


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

for input_words, target_words in zip(input_texts, target_texts):
    encoder_input_wids = []
    for w in input_words:
        w2idx = 1  # default [UNK]
        if w in input_word2idx:
            w2idx = input_word2idx[w]
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


























##########

print("loading data")
X = create_dataset("data/data.in",num_examples)
Y = create_dataset("data/data.ou",num_examples)

'''
def check_sentence(X,Y):
    xn = []
    yn = []
    for i in range(len(X)-1):
        if len(Y[i]) <= 2*len(X[i]) and len(Y[i]) <= 50:
            xn.append(X[i])
            yn.append(Y[i])
    
    return xn,yn
 

X, Y = check_sentence(X,Y)
'''
def max_length(tensor):
    return max(len(t) for t in tensor)






input_tensor, inp_lang = tokenize(tuple(X))
target_tensor, targ_lang = tokenize(tuple(Y))
inp_dict, targ_dict = inp_lang.word_counts, targ_lang.word_counts



def filter_dict(d,frequency):
    d_new = {}
    for k in d:
        if d[k] >= frequency:
            d_new[k] = d[k]
    return d_new
'''
inp_dict_new = {}
for k in inp_dict:
    if inp_dict[k] >= 5:
        inp_dict_new[k] = inp_dict[k]
        

'''
inp_dict_new = filter_dict(inp_dict,5)
targ_dict_new = filter_dict(targ_dict,5)

a = 0
for i in range(len(X)):
    l = X[i].split()
    for j in range(len(l)):
        #print(k)
        if l[j] not in inp_dict_new:
            X[i] = '<unknown>'
            a = a+1
            
    
X = list(map(lambda x: " ".join(x for x in x.split() if x in inp_dict_new),X))
Y = list(map(lambda x: " ".join(x for x in x.split() if x in inp_dict_new),Y))

input_tensor, inp_lang = tokenize(tuple(X))
target_tensor, targ_lang = tokenize(tuple(Y))


#input_tensor, inp_lang, input_mat = tokenize(tuple(X))
#target_tensor, targ_lang, target_mat = tokenize(tuple(Y))


#input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

#max_length_inp,max_length_targ = max_length(input_tensor), max_length(target_tensor)


encoder_input_data = input_tensor
 




 
print("loading data complete. Now create model!")



#############################################
#### Train the model and tuning the parameters 
BATCH_SIZE = 50
NUM_EPOCHS = 1000
HIDDEN_UNITS = 256  
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
optimizer = 'adam'
#optimizer = 'rmsprop'



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
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(encoder_input_data, target_texts, test_size=0.2, random_state=42)
    
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
train(NUM_EPOCHS,BATCH_SIZE,optimizer)














 




 




