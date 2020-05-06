#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.24 
# First create: 2019.03.24 
# Description:
# check the content of labels 



import os 
import sys 
import numpy as np 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)




def collection_labels(repo_path, file_name):
    data_path = os.path.join(repo_path, file_name)
    with open(data_path, "r") as f:
        data_items = f.readlines()
        label_collections = []
        for data_item in data_items:
            data_item = data_item.strip()
            if len(data_item) == 0:
                continue 
            label_item = data_item.split(" ")
            # print("check the content of label_item")
            # print(label_item)
            label_collections.append(label_item[1])

        print("check the full labels of text is : ")
        print(set(label_collections))



if __name__ == "__main__":
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/MSRANER"
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/OntoNote4NER" 
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/ResumeNER" 
    # file_name = "train.char.bmes"
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/CTB5POS" 
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/CTB6POS" 
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/UD1POS" 
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/PKUCWS" 
    # repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/CTB6CWS" 
    repo_path = "/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/MSRCWS"
    file_name = "train.char.bmes" 
    collection_labels(repo_path, file_name)
