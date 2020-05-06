#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.02 
# First create: 2019.04.02 
# Description:
# select the NON-tagging labels 
# /data/nfsdata/xiaoya/datasets/zh_tagging/MSRANER



import os 
import sys 
import numpy as np 



root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path) 



def read_lines(data_repo):
    with open(data_repo, "r") as f:
        data_lines = f.readlines() 
    return data_lines 



def rewrite_dataset(data_repo):
    dev_path = os.path.join(data_repo, "dev.char.bmes_bk")
    test_path = os.path.join(data_repo, "test.char.bmes_bk") 
    dev_lines = read_lines(dev_path)
    test_lines = read_lines(test_path) 

    with open(os.path.join(data_repo, "dev.char.bmes"), "w") as f:
        for dev_item in dev_lines:
            data_item = dev_item.split("\t")[1] 
            if "B" in data_item or "S" in data_item:
                f.write(dev_item) 

    with open(os.path.join(data_repo, "test.char.bmes"), "w") as f:
        for test_item in test_lines:
            data_item = test_item.split("\t")[1]
            if "B" in data_item or "S" in data_item:
                f.write(test_item) 

if __name__ == "__main__":
    data_repo = "/data/nfsdata/xiaoya/datasets/zh_tagging/MSRANER" 
    rewrite_dataset(data_repo) 
