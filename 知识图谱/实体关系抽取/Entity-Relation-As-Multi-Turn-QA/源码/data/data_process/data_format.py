#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.23 
# First cretae: 2019.03.23 
# Description:
# data_format_change.py


import os 
import sys 
import numpy as np



root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)



def load_dataset(data_path):
    # load dataset into memory from text file 
    dataset = []
    with open(data_path, "r") as f:
        words, tags = [], []
        # each line of the file corresponds to one word and tag 
        cnt = 0
        for line in f:
            cnt += 1
            if line != "\n":
                line = line.strip()
                #try:
                word, tag = line.split(" ")
                #except Exception as e:
                #    print(data_path)
                #    print(cnt)
                #    word = ' '
                #    print(word)
                #    print(tag)
                # tag = tag.replace('W', 'SEG')
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print("an exception was raised , skipping a word ")
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset 



def dump_dataset(data_lines, data_path):
    print("dataset format of the file is TSV")
    with open(data_path, "w") as f:
        for data_item in data_lines:
            data_word, data_tag = data_item
            data_str = " ".join(data_word)
            data_tag = " ".join(data_tag)
            f.write(data_str+ "\t" + data_tag + "\n")
        print("dump data set int data path")
        print(data_path)


def main(source_repo_path, target_repo_path):
    print("please notice that the file of MSRA NER is as follows")
    source_train_path = os.path.join(source_repo_path, "train.txt")
    target_train_path = os.path.join(target_repo_path, "train.char.bmes")
    train_set = load_dataset(source_train_path)
    dump_dataset(train_set, target_train_path)

    source_dev_path = os.path.join(source_repo_path, "dev.txt")
    target_dev_path = os.path.join(target_repo_path, "dev.char.bmes")
    dev_set = load_dataset(source_dev_path)
    dump_dataset(dev_set, target_dev_path)


    source_test_path = os.path.join(source_repo_path, "test.txt")
    target_test_path =  os.path.join(target_repo_path, "test.char.bmes")
    test_set = load_dataset(source_test_path)
    dump_dataset(test_set, target_test_path)



if __name__ == "__main__":
    # need 
    task_target = "good_as"
    source_repo = "/data/nfsdata/xiaoya/datasets/zh_tagging/cws/datas/"
    source_repo_path = os.path.join(source_repo, task_target)
    target_repo = "/data/nfsdata/xiaoya/datasets/zh_tagging/cws/datas"
    target_repo_path = os.path.join(target_repo, task_target)
    main(source_repo_path, target_repo_path)
