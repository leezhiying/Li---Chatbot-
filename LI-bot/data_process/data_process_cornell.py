#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:16:30 2019

@author: lizhiying
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import re
import operator


os.chdir('/Users/lizhiying/Desktop/Big Data Analytics/LI-bot')


DATA_PATH = 'data/'
CPT_DIR = 'checkpoints/'
MOVIE_LINES_FILE = 'movie_lines.txt'
MOVIE_CONVOS_FILE = 'movie_conversations.txt'
VOCAB_COUNT_THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]

# The following functions are used to retrieve the
# Cornell Movie-Dialogs Corpus dataset


def get_lineid_content():
    """ Get the pair of lineID and content
        movie_lines.txt file includes the corresponding
        conversation content to each lineID
    """
    lineid_content = {}
    lines_file_path = os.path.join(DATA_PATH + MOVIE_LINES_FILE)

    with open(lines_file_path, 'r', errors='ignore') as f:
        # +++$+++ is used to split the section in a single line
        # A correct formed line includes five sections
        # The first section is lineID
        # The last section is line content
        # Here we only need lineID and content

        for line in f:
            line_sections = line.split(' +++$+++ ')
            assert len(line_sections) == 5
            if line_sections[4][-1] == '\n':
                line_sections[4] = line_sections[4][:-1]
            lineid_content[line_sections[0]] = line_sections[4]

    return lineid_content


def get_convos():
    """ Reconstruct the movie conversation into pair of dialog
        into list of ordered lineIDs
    """
    convos = []
    convos_file_path = os.path.join(DATA_PATH, MOVIE_CONVOS_FILE)

    with open(convos_file_path, 'r', errors='ignore') as f:
        # +++$+++ is used to split the section in a single line
        # A correct formed line includes four sections
        # The last section is list of lineIDs in each conversation

        for line in f:
            line_sections = line.split(' +++$+++ ')
            assert len(line_sections) == 4
            convos.append(line_sections[3][1:-2].replace('\'', '').split(', '))

    return convos


def get_data(lineid_content, convos):
    # Construct the corresponding input and output conversation pair
    # Turn lineID into content

    input_set, output_set = [], []
    for each_convo in convos:
        for index, lineid in enumerate(each_convo[:-1]):
            # Input and output data should be roughly less than 50 words
            # And we hope that output length is less than 2 times of input
            # Cause when output length is bigger than 2 times of input
            # There's highly chance that it makes no sense to infer the output

            input_length = len(
                lineid_content[each_convo[index]].split(' '))
            output_length = len(
                lineid_content[each_convo[index + 1]].split(' '))

            if input_length < 50 and output_length < 50 and \
                    output_length < 2 * input_length:
                input_set.append(lineid_content[each_convo[index]])
                output_set.append(lineid_content[each_convo[index + 1]])
            else:
                continue

    # Make sure than the length of input_set is equal to output_set
    # Since every input sentence corresponds to one output response
    assert len(input_set) == len(output_set)
    
    # Split the pair into train set and test set
    # Train set : Dev set : Test set = 7 : 2 : 1
    filenames = ['cornell.in', 'cornell.ou']
    file_pool = []
    for each_file in filenames:
        file_pool.append(open(os.path.join(DATA_PATH, each_file), 'w+'))
        
    for i in range(len(input_set)):
        file_pool[0].write(input_set[i]+'\n')
        file_pool[1].write(output_set[i]+'\n')
    
    for file in file_pool:
        file.close

def data_preprocessing():
    """ Data preprocessing procedure, mainly for process
        Cornell Movie-Dialogs Corpus
    """
    lineid_content = get_lineid_content()
    print('Read movie_lines.txt file complete...')
    convos = get_convos()
    print('Read movie_conversations.txt file complete...')
    print('Building dataset')
    get_data(lineid_content, convos)

if __name__ == '__main__': 
    data_preprocessing()
    
    