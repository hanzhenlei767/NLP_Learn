#!/usr/bin/python
# coding=utf8

"""
# Created : 2020/10/22
# Version : python3.6
# Author  : hzl 
# File    : tool_file.py
# Desc    : 读写文本文件,pickle,json,glove,stopwords等方法
"""

import os
import pickle as pkl
import json

class File:
    def __init__(self):
        pass

    #text读写操作
    def read_text(self,text_path):
        with open(text_path,'r',encoding='utf-8') as fr:
            lines = fr.readlines()
            text_list = [line.strip() for line in lines]
        return text_list

    def write_text(self,string,text_path):
        with open(text_path,'w',encoding='utf-8') as fw:
            fw.write(string+"\n")


    #pickle读写操作
    def load_pkl(self, file_path):
        with open(file_path, 'rb') as fl:
            obj = pkl.load(fl)
        return obj

    def dump_pkl(self, obj, file_path):
        with open(file_path, 'wb') as fd:
            pkl.dump(obj, fd)


    #json读写操作
    def read_json(self,json_path):
        with open(json_path,'r',encoding='utf-8') as fr:
            json_dic = json.load(fr)
        return json_dic

    def save_json(self,json_dic,json_path):
        with open(json_path,'w',encoding='utf-8') as fw:
            json.dump(json_dic, fw, indent=4)


    #读glove词向量
    def read_glove(self,glove_path):
        w2v = {}
        with open(file=glove_path, encoding="UTF-8", mode="r") as fr:
            lines = fr.readlines()
            for line in lines:
                seg_line = line.strip().split()
                word = seg_line[0]
                w2v[word] = list(map(lambda x: float(x),seg_line[1:]))
        return w2v


    #读stopwords
    def read_word(self,word_path):
        with open(word_path,'r',encoding='utf-8') as fr:
            lines = fr.readlines()
            words_list = [line.strip() for line in lines]
        return words_list





