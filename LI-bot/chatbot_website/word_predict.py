#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:46:18 2019

@author: lizhiying
"""
import os 
os.chdir('/Users/lizhiying/Desktop/Big Data Analytics/LI-bot')

from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk

HIDDEN_UNITS = 200
#HIDDEN_UNITS = 256


class CornellWordChatBot(object):
    model = None
    encoder_model = None
    decoder_model = None
    input_word2idx = None
    input_idx2word = None
    target_word2idx = None
    target_idx2word = None
    max_encoder_seq_length = None
    max_decoder_seq_length = None
    num_encoder_tokens = None
    num_decoder_tokens = None

    def __init__(self):
        self.input_word2idx = np.load('models/word-lstm/word-input-word2idx.npy').item()
        self.input_idx2word = np.load('models/word-lstm/word-input-idx2word.npy').item()
        self.target_word2idx = np.load('models/word-lstm/word-target-word2idx.npy').item()
        self.target_idx2word = np.load('models/word-lstm/word-target-idx2word.npy').item()
        context = np.load('models/word-lstm/word-context.npy').item()
        self.max_encoder_seq_length = context['encoder_max_seq_length']
        self.max_decoder_seq_length = context['decoder_max_seq_length']
        self.num_encoder_tokens = context['num_encoder_tokens']
        self.num_decoder_tokens = context['num_decoder_tokens']

        encoder_inputs = Input(shape=(None, ), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.num_encoder_tokens, output_dim=HIDDEN_UNITS,
                                      input_length=self.max_encoder_seq_length, name='encoder_embedding')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm")
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # model_json = open('../chatbot_train/models/cornell/word-architecture.json', 'r').read()
        # self.model = model_from_json(model_json)
        self.model.load_weights('models/word-lstm/word-weights.h5')
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def reply(self, input_text):
        input_seq = []
        input_wids = []
        for word in nltk.word_tokenize(input_text.lower()):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
            
            
            
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_encoder_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()

    def test_run(self):
        while True:
            chat = input("Let's chat \r")
            print(self.reply(chat))
            if chat ==  "Q":
                break

        

def main():
    model = CornellWordChatBot()
    model.test_run()
'''
if __name__ == '__main__':
    main()





from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
model = CornellWordChatBot()





input_path = "data/cornell.in"
target_path = "data/cornell.ou"
lines_input = open(input_path, 'rt', encoding='utf8').read().split('\n')
lines_target = open(target_path, 'rt', encoding='utf8').read().split('\n')
cornell_input = []
cornell_target = []


for line in lines_input:
    cornell_input.append(line)

for line in lines_target:
    cornell_target.append(line)


input_path = "data/squad.in"
target_path = "data/squad.ou"
lines_input = open(input_path, 'rt', encoding='utf8').read().split('\n')
lines_target = open(target_path, 'rt', encoding='utf8').read().split('\n')
squad_input = []
squad_target = []


for line in lines_input:
    squad_input.append(line)

for line in lines_target:
    squad_target.append(line)


input_texts = cornell_input + squad_input
target_texts = cornell_target + squad_target

input_texts = input_texts[1:1000]
target_texts = target_texts[1:1000]

score = 0
for i in range(len(input_texts)-1):
    print(i)
    reply = model.reply(input_texts[i])
    reply = [w.lower() for w in nltk.word_tokenize(reply)]
    truth = [w.lower() for w in nltk.word_tokenize(target_texts[i])]
    s = sentence_bleu(reply,truth,weights = (0.5,0.5,0,0))
    print(s)
    score += s 
    
score = score/len(input_texts)

'''



