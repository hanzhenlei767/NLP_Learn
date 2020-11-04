#!/usr/bin/python
# coding=utf8

"""
# Created : 2020/10/22
# Version : python3.6
# Author  : hzl 
# File    : tool_rename.py
# Desc    : 修改指定文件夹下指定文件类型到指定指定文件夹下的指定文件类型
"""
import os
import platform

class Rename:
    def __init__(self):
        pass

    def re_name(self, work_dir, target_dir, old_ext, new_ext):
        """
        传递当前目录，原来后缀名，新的后缀名后，批量重命名后缀
        """
        for filename in os.listdir(work_dir):
            # 获取得到文件后缀
            split_file = os.path.splitext(filename)
            #print(split_file)
            file_ext = split_file[1]
            #print(file_ext)
            # 定位后缀名为old_ext 的文件
            if old_ext == file_ext:
                #return None
                # 修改后文件的完整名称
                newfile = split_file[0] + new_ext
                
                if not os.path.isdir(target_dir):
            
                    os.makedirs(target_dir)

                sys = platform.system()
                # 实现重命名操作
                if sys == "Windows":
                    command_str = "copy " + '"' + os.path.join(work_dir, filename) + '"'+ " " + '"' + os.path.join(target_dir, filename) + '"'
                    #print("OS is Windows!!!")
                elif sys == "Linux":
                    command_str = "cp " + '"' + os.path.join(work_dir, filename) + '"'+ " " + '"' + os.path.join(target_dir, filename) + '"'
                    #print("OS is Linux!!!")
                else:
                    pass
                #print(command_str)
                os.system(command_str) 

                os.rename(
                    os.path.join(target_dir, filename),
                    os.path.join(target_dir, newfile)
                )
        print("重命名完成")



