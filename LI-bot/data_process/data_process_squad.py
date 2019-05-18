#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:22:22 2019

@author: lizhiying
"""

import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

DATA_PATH = 'data/'




os.chdir("/Users/lizhiying/Desktop/Big Data Analytics/LI-bot")
with open("data/train-v2.0.json") as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']

input_set = []
output_set = []


for article in dataset:
    for p in article['paragraphs']:
        for qa in p['qas']:
            if qa['is_impossible'] == False:
                input_set.append(qa['question'])
                output_set.append(qa['answers'][0]['text'])
 

filenames = ['squad.in', 'squad.ou']
file_pool = []

for each_file in filenames:
    file_pool.append(open(os.path.join(DATA_PATH, each_file), 'w+'))


for i in range(len(input_set)):
    file_pool[0].write(input_set[i]+'\n')
    file_pool[1].write(output_set[i]+'\n')

for file in file_pool:
    file.close

    






