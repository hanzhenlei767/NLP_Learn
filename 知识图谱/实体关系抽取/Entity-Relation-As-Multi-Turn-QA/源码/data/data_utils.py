#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.24 
# First create: 2019.03.24 
# Description:
# BERT finetuning runner 
# data_utils.py 


import os 
import sys 
import csv 
import logging 
import argparse 
import random 
import numpy as np 
from tqdm import tqdm, trange 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler 

from data.apply_text_norm import process_sent 


class InputExample(object):
    # a single training / test example for simple sequence classification 
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Construct s input Example. 
        Args:
            guid: unqiue id for the example.
            text_a: string, the untokenzied text of the first seq. for single sequence 
                tasks, only this sequction msut be specified. 
            text_b: (Optional) string, the untokenized text of the second sequence. 
            label: (Optional) string, the label of the example, This should be specifi
                for train and dev examples, but not for test examples. 
        """
        self.guid = guid 
        self.text_a = text_a 
        self.text_b = text_b 
        self.label = label 



class InputFeature(object):
    # a single set of features of data 
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids 
        self.input_mask = input_mask 
        self.segment_ids = segment_ids 
        self.label_id = label_id 



class DataProcessor(object):
    # base class for data converts for sequence classification datasets 
    def get_train_examples(self, data_dir):
        # get a collection of "InputExample" for the train set 
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        # gets a collections of "InputExample" for the dev set 
        raise NotImplementedError()

    def get_labels(self):
        # gets the list of labels for this data set 
        raise NotImplementedError()

    @classmethod 
    def _read_tsv(cls, input_file, quotechar=None):
        # reads a tab separated value file. 
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines 


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}
    # print("check the content of labels") 
    # print(label_map)
    # exit()

    # print("=*="*10) 
    # print("check label list")
    # print(label_list) 
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        # tokens_a = tokenizer.tokenize(process_sent(example.text_a))
        # tokens_a = list(process_sent(example.text_a))  
        # tokens_a = list(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # print("check the content of tokens")
        # print(tokens) 
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # zero padding up to the sequence length 
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding 
        input_mask += padding 
        segment_ids += padding 

        assert len(input_ids) == max_seq_length 
        assert len(input_mask) == max_seq_length 
        assert len(segment_ids) == max_seq_length 
 
        if len(example.label) > max_seq_length - 2:
            example.label = example.label[: (max_seq_length - 2)] 

        label_id = [label_map["O"]] + [label_map[tmp] for tmp in example.label] + [label_map["O"]] 
        label_id += (len(input_ids)-len(label_id)) * [label_map["O"]]
        # print("-*-"*10)
        # print("check the content and length of labels and texts")
        # print(len(input_ids), len(label_id)) 
        # print("-*-"*10) 
        # label_map[example.label]
        
        features.append(InputFeature(input_ids=input_ids, \
            input_mask=input_mask, segment_ids=segment_ids, \
            label_id=label_id))

    return features 
