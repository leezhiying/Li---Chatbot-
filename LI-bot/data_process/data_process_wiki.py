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
import nltk 

os.chdir("/Users/lizhiying/Desktop/Big Data Analytics/LI-bot")
DATA_PATH = 'data/'

input_set = []
output_set = []


with open("data/WikiQA-train.txt") as f:
    q = f.readlines()

all_set = []

for item in q:
    if item[-2] == "1":
        all_set.append(item)



for item in all_set:
    item_l = item.split('\t')
    input_set.append(item_l[0])
    output_set.append(item_l[1])




filenames = ['wiki.in', 'wiki.ou']
file_pool = []

for each_file in filenames:
    file_pool.append(open(os.path.join(DATA_PATH, each_file), 'w+'))
    
for i in range(len(input_set)):
    file_pool[0].write(input_set[i]+'\n')
    file_pool[1].write(output_set[i]+'\n')

for file in file_pool:
    file.close