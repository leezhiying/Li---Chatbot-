#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:36:13 2019

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
import string 
random.seed(0)

#################################################################################
print("loading data")
MAX_INPUT_SEQ_LENGTH = 200
MAX_TARGET_SEQ_LENGTH = 200
MAX_VOCAB_SIZE = 2000
#NUM_SAMPLE = 10000


def not_strange(char):
    if char in string.ascii_letters or char in string.digits or char in string.punctuation or char == ' ':
        return True 
    else:
        return False


def load_data(input_path,target_path):
    lines_input = open(input_path, 'rt', encoding='utf8').read().split('\n')
    lines_target = open(target_path, 'rt', encoding='utf8').read().split('\n')
    input_texts = [ ]
    target_texts = [ ]

    for (line_input,line_target) in zip(lines_input,lines_target):
        c_input = True
        c_target = True 
        
        for item in line_input:
            if not_strange(item):
                c_input = True
            else:
                c_input = False 
                break 
            
        for item in line_target:
            if not_strange(item):
                c_target = True
            else:
                c_target = False 
                break 
 
        if c_input==True and c_target == True and len(line_input) < MAX_INPUT_SEQ_LENGTH and len(line_target) < MAX_TARGET_SEQ_LENGTH:
            input_texts.append(line_input)
            line_target = '\t' + line_target + '\n'
            target_texts.append(line_target)
    
    return input_texts, target_texts 


cornell_input , cornell_output = load_data("data/cornell.in","data/cornell.ou")
squad_input, squad_output = load_data("data/squad.in","data/squad.ou")
wiki_input, wiki_output = load_data("data/wiki.in","data/wiki.ou")


input_texts = cornell_input + squad_input + wiki_input
target_texts = cornell_output + squad_output + wiki_output

input_characters = set()
target_characters = set()



print(input_texts[8883],target_texts[8883])


for input_text, target_text in zip(input_texts, target_texts):
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
 
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_char2idx = dict([(char, i) for i, char in enumerate(input_characters)])
input_idx2char = dict([(i, char) for i, char in enumerate(input_characters)])
target_char2idx = dict([(char, i) for i, char in enumerate(target_characters)])
target_idx2char = dict([(i, char) for i, char in enumerate(target_characters)])

np.save('models/char-lstm/char-input-char2idx.npy', input_char2idx)
np.save('models/char-lstm/char-target-char2idx.npy', target_char2idx)
np.save('models/char-lstm/char-input-idx2char.npy', input_idx2char)
np.save('models/char-lstm/char-target-idx2char.npy', target_idx2char)

context = dict()
context['max_encoder_seq_length'] = max_encoder_seq_length
context['max_decoder_seq_length'] = max_decoder_seq_length
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens

np.save('models/char-lstm/char-context.npy', context)



#############################################################################################################
print("data loaded. now train the model")
BATCH_SIZE = 64
NUM_EPOCHS = 1
HIDDEN_UNITS = 256
WEIGHT_FILE_PATH = 'models/char-lstm/char-weights.h5'

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_char2idx[char]] = 1
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_char2idx[char]] = 1
        if t > 0:
            decoder_target_data[i, t-1, target_char2idx[char]] = 1


encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_inputs')
encoder = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print(model.summary())

#json = model.to_json()
#open('models/char-lstm/char-architecture.json', 'w').write(json)

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
          validation_split=0.2, callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)
