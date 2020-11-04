#!/usr/bin/python
# coding=utf8

"""
# Created : 2020/10/22
# Version : python3.6
# Author  : hzl 
# File    : tool_url.py
# Desc    : 日志打印在控制台和文本文件中
"""
import os
import win32com.client as win32
from xlutils.copy import copy
import xlrd, xlwt
import shutil

class Url:
    def __init__(self):
        pass

    #xlsx和xls格式互转
    def xlsx_to_xls(self, src, dest):
        excel=win32.gencache.EnsureDispatch('excel.application')
        pro=excel.Workbooks.Open(src)   #打开要转换的excel
        pro.SaveAs(dest,FileFormat=56)  #另存为xls格式
        pro.Close()
        excel.Application.Quit()

    def xls_to_xlsx(self, src, dest):
        excel=win32.gencache.EnsureDispatch('excel.application')
        pro=excel.Workbooks.Open(src)     #打开要转换的excel
        pro.SaveAs(dest,FileFormat=51)     #另存为xls格式
        pro.Close()
        excel.Application.Quit()


    def play_to_url(self, work_dir, target_dir):

        filename_list = os.listdir(work_dir)  #工作目录下所有的文件
        print(filename_list)

        #过滤正在打开的文件
        filename_list = [filename for filename in filename_list if not filename.startswith('~')]
        print(filename_list)

        split_file = [os.path.splitext(filename) for filename in filename_list]

        file_list = [file for file in split_file if file[1] == '.xlsx']

        for file in file_list:
            #xlsx转xls
            tranfile1 = os.path.join(work_dir, file[0]+file[1]) #要转换的excel
            tranfile2 = os.path.join(work_dir, file[0]+".xls")  #转换出来excel

            #xlsx转换成xls
            self.xlsx_to_xls(tranfile1, tranfile2)

            shutil.rmtree(tranfile1)

            #修改超链接
            m = xlrd.open_workbook(tranfile2)
            ls = m.sheet_names()
            sheet_1 = m.sheet_by_name(ls[0])
            col,row = sheet_1.ncols,sheet_1.nrows
            print(row,col)
            print(type(sheet_1))
            # 将操作文件对象拷贝，变成可写的workbook对象
            new_excel = copy(m)

            # 获得第一个sheet的对象
            ws = new_excel.get_sheet(0)
            print(type(ws))
            ws.write(0, 11, '超链接')

            for j in range(1,row):
                link = sheet_1.hyperlink_map.get((j,11))
                ws.write(j, 11, '(No URL)' if link is None else link.url_or_path)

            new_excel.save(tranfile2)

            #xls转xlsx
            self.xls_to_xlsx(tranfile2,tranfile1)

            os.remove(tranfile2)






















