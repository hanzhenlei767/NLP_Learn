#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last upate: 2019.03.22 
# First create: 2019.03.22 
# Description:
# glyph_embedding_config.py



import os
import sys 
import json 
import copy 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)




class GlyceEmbeddingConfig:
    def __init__():
        self.dropout = dropout 
        self.idx2word = idx2word 
        self.idx2char = idx2char 

        self.word_embsize = word_embsize 
        self.glyph_embsize = glyph_embsize 
        self.pretrained_char_embedding = None 
        self.pretrained_word_embedding = None 
        self.font_channels = font_channels 
        self.random_fonts = random_fonts 
        self.font_name = font_name 
        self.font_size = font_size 
        self.use_tradiational = use_tranditional 
        self.font_normalize = font_normalize 
        self.subchar_type = subchar_type 
        self.subchar_embsize = subchar_embsize 

        self.random_erase = random_erase 
        self.num_fonts_concat = num_fonts_concat 
        self.glyph_cnn_type = glyph_cnn_type 
        self.cnn_dropout = cnn_dropout 
        self.use_batch_norm = use_batch_norm 
        self.use_highway = use_highway 
        self.yuxian_merge = yuxuan_merge 
        self.fc_merge = fc_merge 
        self.output_size = output_size 
        self.char_embsize = char_embsize 
        self.level = level 
        self.char_drop = char_drop 
        self.char2word_dim = char2word_dim 

        
