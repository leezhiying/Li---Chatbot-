#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:40:43 2019

@author: lizhiying
"""

#Let's create a chatbot! 

from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

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
from sklearn.model_selection import train_test_split
os.chdir('/Users/lizhiying/Desktop/Big Data Analytics/LI-bot')




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



##########

print("loading data")
X = create_dataset("data/cornell.in",num_examples)
Y = create_dataset("data/cornell.ou",num_examples)

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


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)
  
  tensor = lang_tokenizer.texts_to_sequences(lang)
  
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
  
  return tensor, lang_tokenizer




input_tensor, inp_lang = tokenize(tuple(X))
target_tensor, targ_lang = tokenize(tuple(Y))



input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
#len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)


max_length_inp,max_length_targ = max_length(input_tensor), max_length(target_tensor)

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

#print ("Input index to word mapping")
#onvert(inp_lang, input_tensor_train[10])





 
print("loading data complete. Now create model!")


########hyperparameters####
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 20
N_BATCH = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 10
units = 100
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1


#######create a dataframe######
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)














def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               #recurrent_activation='relu', 
                               recurrent_activation = 'sigmoid',
                               recurrent_initializer='glorot_uniform')





####Build the Encode and Decode model 
print("Generating Encoder and Decoder!")

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)        
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)



 

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))






decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)




#######################Optimizer#############################



optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, epsilon = 1e-08)


def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)



checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)




###########################Training#######################
def run_model():
        
    EPOCHS = 10
    print("The total batch number is ",N_BATCH,"Please wait for a moment")
    
    for epoch in range(EPOCHS):
        start = time.time()
        
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        
        for (batch, (inp, targ)) in enumerate(dataset):
            #print(batch)
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                
                dec_hidden = enc_hidden
                
                dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)       
                
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    
                    loss += loss_function(targ[:, t], predictions)
                    
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            
            total_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            
            gradients = tape.gradient(loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables))
            
            #if batch % 100 == 0:
            print('Epoch {} Batch {} Total Batch {} Loss {:.4f}'.format(epoch + 1,batch,N_BATCH,batch_loss.numpy()))
        # saving (checkpoint) the model every 1 epochs
        if (epoch + 1) % 1 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))








def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot







# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()

def chat(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
    print('Input: {}'.format(sentence))
    print('Output: {}'.format(result))
    
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


























#run_model()




# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



 
chat('I do', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)











