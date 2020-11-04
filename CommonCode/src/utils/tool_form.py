#!/usr/bin/python
# coding=utf8

"""
# Created : 2020/10/22
# Version : python3.6
# Author  : hzl 
# File    : tool_form.py
# Desc    : 读写csv,xls等方法
"""

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import xlrd, xlwt #引入读写工作表的库

class Form:
    def __init__(self):
        pass

    #xlsx多sheet情况读取
    def read_xlsx(self,xls_path,sheet_id):
        xls_file=pd.ExcelFile(xls_path)
        sheet_names_list = xls_file.sheet_names
        print("sheet_id to sheet_name:",sheet_names_list[sheet_id])
        table = xls_file.parse(sheet_names_list[sheet_id])
        return table

    def save_xlsx(self,dataframe,xls_path):
        dataframe.to_excel(xls_path)

    #csv读取保存
    def read_csv(self,csv_path):
        table = pd.read_csv(csv_path,engine='python')
        return table

    def save_csv(self,csv_path):
        pd.read_csv(csv_path)

