#!/usr/bin/python
# coding=utf8

"""
# Created : 2020/10/22
# Version : python3.6
# Author  : hzl 
# File    : tool_log.py
# Desc    : 日志打印在控制台和文本文件中
"""
import logging
import sys
import os

class Log:
    def __init__(self, log_dir,log_name = 'log.txt'):
        '''
        功能：初始化logging
        输入参数：log_dir-日志保存路径
        输入参数：log_name-日志保存文件名
        '''
        self.log_file = os.path.join(log_dir, log_name)
        
        if not os.path.isdir(log_dir):
            
            os.makedirs(log_dir)

        self.log = logging.getLogger()
        
        fh = logging.FileHandler(self.log_file)
        
        formatter = logging.Formatter(fmt="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", datefmt='%Y/%m/%d %I:%M:%S')

        fh.setFormatter(formatter)
        
        self.log.addHandler(fh)
        
        self.log.setLevel(logging.INFO)
        
        #在控制台输出日志
        sh = logging.StreamHandler(sys.stdout)
        
        self.log.addHandler(sh)
        
    def __call__(self, *args):
        '''
        功能：魔术函数-可使对象直接传参调用
        '''
        temp = ""
        for i in args:
            temp += i
            temp += ","
        self.log.info(temp[:-1]) 

    def newline(self):
        '''
        功能:打印空格
        '''
        self.log.info('')




