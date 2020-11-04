#!/usr/bin/python
# coding=utf8

"""
# Created : 2020/10/22
# Version : python3.6
# Author  : hzl 
# File    : tool_drawing.py
# Desc    : 画饼图、散点图、柱状图方法
"""

import os
import numpy as np
import matplotlib.pyplot as plt

class Drawing:
    def __init__(self):
        pass

    #画饼图
    def draw_pie(self,labels,sizes,title='',save_path=''):
        '''
        labels:饼图扇区名
        sizes:饼图各扇区比例
        title:饼图的标题
        save_path:Pic保存路径。默认不保存，保存文件以title为文件名
        '''
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False#用于正常显示负号

        #colors='lightgreen','gold','lightskyblue','lightcoral','pink'#颜色

        explode=[0.1]*len(labels)#扇区中间的间隔

        plt.pie(x=sizes ,#扇区的面积
                explode=explode,#扇区中间的间隔
                labels=labels,#饼图每个扇区的名字
                colors=None,#饼图每个扇区的颜色
                autopct='%1.1f%%',#每个扇区所占比例的显示格式
                shadow=True,#在饼图下面画一个阴影。默认值：False，即不画阴影；
                startangle=50,#起始绘制角度,默认图是从x轴正方向逆时针画起,如设定=90则从y轴正方向画起；
                counterclock=True,#指定指针方向；布尔值，可选参数，默认为：True，即逆时针。将值改为False即可改为顺时针。
                labeldistance = 1.1,#label绘制位置,相对于半径的比例, 如<1则绘制在饼图内侧，默认值为1.1
                pctdistance=0.6#类似于labeldistance,指定autopct的位置刻度，默认值为0.6
                )

        plt.axis('equal')#该行代码使饼图长宽相等

        plt.legend()#添加图例

        if title:
            plt.title(title)

        if save_path:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, title+"pie"+".png"),
                        dpi=200,
                        bbox_inches='tight')#保存图表

        plt.show()

    #画散点图
    def draw_scatter(self,x,y,side=2,copies=4,xlabel='pred',ylabel='label',title='',save_path=''):
        '''
        x:散点图横坐标
        y:散点图纵坐标
        title:散点图标题
        save_path:Pic保存路径。默认不保存，保存文件以title为文件名
        '''
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False #用于正常显示负号
        #colors='lightgreen','gold','lightskyblue','lightcoral','pink'#颜色
        
        fig = plt.figure(figsize=(10, 10))

        plt.xlim(0, side)#设置x轴的取值范围为：0到2
        plt.ylim(0, side)#设置y轴的取值范围为：0到2

        plt.xlabel(xlabel)  #设置X轴的标题
        plt.ylabel(ylabel) #设置Y轴的标题

        #plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
        plt.grid(linestyle='-.')
        plt.xticks(np.arange(0,side+side/copies,side/copies),np.arange(0,side+side/copies,side/copies))
        plt.yticks(np.arange(0,side+side/copies,side/copies),np.arange(0,side+side/copies,side/copies))
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(np.around(x,2),y,c = "r",marker = ".")

        ax.scatter(np.arange(0,side+0.01,0.01),np.arange(0,side+0.01,0.01)+side*0.2,c = "y",marker = ".")
        ax.scatter(np.arange(0,side+0.01,0.01),np.arange(0,side+0.01,0.01)-side*0.2,c = "y",marker = ".")

        if title:
            plt.title(title)

        if save_path:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, title+"scatter"+".png"),
                        dpi=200,
                        bbox_inches='tight')#保存图表

        plt.show()



    #画柱状图
    def draw_bar(self,x,y,width=0.05,title='',save_path=''):
        '''
        x:x坐标
        y:柱状图条形高度
        title:柱状图标题
        save_path:Pic保存路径。默认不保存，保存文件以title为文件名
        '''
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False#用于正常显示负号

        plt.bar(x=x,                    #x坐标
                height=y,               #条形的高度
                width=width,             #条形的宽度0~1，默认0.8
                #botton=None,           #条形的起始位置
                alpha=0.5,              #柱体填充颜色的透明度
                align='center',         #x轴上的坐标与柱体对其的位置,'edge'
                color='gold',           #“r","b","g",默认“b"
                edgecolor='red',        #边框的颜色
                linewidth=1.5,          # 柱体边框线的宽度
                label="人数",           #柱子的含义
                tick_label=x,           #每个柱体的标签名称
                orientation="vertical"  #竖直："vertical"，水平条："horizontal"
                )

        #柱状图上标文字
        for a,b in zip(x,y):
            plt.text(a, b+0.1, b, ha='center', va='bottom')

        plt.legend(loc="upper center")                  #添加图例的位置
        plt.rcParams['font.sans-serif'] = ['SimHei']    #用来正常显示中文标签
        plt.ylabel('样本个数')
        plt.xlabel('分数')
        plt.rcParams['savefig.dpi'] = 100               #图片像素
        plt.rcParams['figure.dpi'] = 100                #分辨率
        #plt.rcParams['figure.figsize'] = (15.0, 8.0)   #尺寸

        if title:
            plt.title(title)

        if save_path:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, title+"bar"+".png"),
                        dpi=200,
                        bbox_inches='tight')#保存图表

        plt.show()


