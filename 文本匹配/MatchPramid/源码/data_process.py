import os
import numpy as np
import pandas as pd
import tensorflow as tf
import jieba
import re
import copy
import pickle
import json
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
from gensim.models import KeyedVectors
 
def clean_str(text):
    text = str(text)
    text = re.sub(r"\'s", " is", text)#有出错的case,:it not only our country\'s droom but also is our everyone\'s
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"'t", " not ", text)
    text = re.sub(r"'m", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    #连续多个句号替换成一个空格
    
    connect_list = re.findall(r"\.\.+[A-Z]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"[A-Z]", i, flags=0)
        text = text.replace(i," . "+second[0])
    
    connect_list = re.findall(r"\.\.+\s[A-Z]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"[A-Z]", i, flags=0)
        text = text.replace(i," . "+second[0])
        
    connect_list = re.findall(r"\.\.+\s[a-z0-9]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"\s[a-z0-9]", i, flags=0)
        text = text.replace(i,second[0])
        
    connect_list = re.findall(r"\.\.+[a-z0-9]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"[a-z0-9]", i, flags=0)
        text = text.replace(i," "+second[0])
    
    #标点前后插入空格
    text = text.replace("?"," ? ")
    text = text.replace(","," , ")
    text = text.replace("."," . ")
    text = text.replace("!"," ! ")


    #小写单词和大写单词连一块的拆分
    connect_list = re.findall(r"\s[a-z]+[A-Z][a-z]*", text, flags=0)
    for i in connect_list:
        first = re.match(r"^[a-z]*", i[1:], flags=0)
        second = re.findall(r"[A-Z][a-z]*", i[1:], flags=0)
        text = re.sub(i, " "+ first.group() + " . " + second[0], text)

    #两个开头大写的单词连一块的拆分： MadamI'm
    connect_list = re.findall(r"\s[A-Z][a-z]+[A-Z][a-z]*", text, flags=0)
    for i in connect_list:
        first = re.match(r"^[A-Z][a-z]*", i[1:], flags=0)
        second = re.findall(r"[A-Z][a-z]*", i[1:], flags=0)
        text = re.sub(i, " "+ first.group() + " . " + second[1], text)


    #文章开头乱码去除
    pattern = r"[A-Z][a-z]+"
    pattern = re.compile(pattern) 
    res = pattern.search(text)
    if res:
        text = text[res.start():]

    #去除识别出来的噪声：Dear Sir or Madam, - I am Li Hua,
    text = re.sub(r"-", " ", text)

    #乱码符号去除
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=\s\"\?]", "", text)

    #多个空格替换成一个空格
    text = re.sub(r"\s+", " ", text)

    #两个小写单词连一块拆分：manysomething ，schoolabout 

    #一个完整单词识别成了两个：fri -endships，be a go od journlist

    #将文本转成小写
    text = text.lower()
    return text


 
# 文件路径
path = 'data/'
original_path = path+'original_data/'
assist_path = path+'assist_data/'
mid_path = path+'mid_data/'

#训练数据
train_data_path = original_path+'train_data.csv'
dev_data_path = original_path+'dev_data.csv'
test_data_path = original_path+'test_data.csv'


#训练数据的分析的外部数据
google_word2vec_path = assist_path+'GoogleNews-vectors-negative300.bin'


def read_data(data_path):
    #读取训练数据
    data_frame = pd.read_csv(data_path, header=0, names=['location','result','fw1','fw2', 'tt_label', 'dd_label'])
    return data_frame

train_data = read_data(train_data_path)
dev_data = read_data(dev_data_path)
test_data = read_data(test_data_path)

print("train data shape:")
print(train_data.shape)
print(train_data.head())
print("dev data shape:")
print(dev_data.shape)
print(dev_data.head())
print("test data shape:")
print(test_data.shape)
print(test_data.head())

#处理数据
def preprocess_data(data_all):
    data_processed = copy.deepcopy(data_all)
    # 加载停用词
    stopwords_list = stopwords.words('english')

    for index, row in data_all.iterrows():
        for col_name in ['result','fw1','fw2']:
            # 分词+去除停用词
            text = clean_str(row[col_name])
            text_list = nltk.word_tokenize(text)

            filtered = [w for w in text_list if(w not in stopwords_list)]
            seg_str = " ".join(filtered)
            #分词之后的DataFrame
            data_processed.at[index, col_name] = seg_str
    return data_processed

train_data_processed = preprocess_data(train_data)
dev_data_processed = preprocess_data(dev_data)
test_data_processed = preprocess_data(test_data)

'''
def merge_words(train_all):
    train_data_process = copy.deepcopy(train_all)
    print('process_embedding process...')
    texts = []
    texts.extend(train_data_process['result'].tolist())
    texts.extend(train_data_process['fw1'].tolist())
    texts.extend(train_data_process['fw2'].tolist())
    print(len(texts))
    return texts


train_texts = merge_words(train_data_processed)
dev_texts = merge_words(dev_data_processed)
test_texts = merge_words(test_data_processed)

texts = []
texts.extend(train_texts)
texts.extend(dev_texts)
texts.extend(test_texts)

def vocab_build(texts,min_count = -float("inf")):
    """
    :param min_count: 最小词频
    :return:  word2id = {'<PAD>':0, 'word1':id_1, ……， '<UNK>':id_n}
    """
    word2id_ct = {}
    for word_list in texts:
        for word in word_list.split():
            if word not in word2id_ct:
                word2id_ct[word] = [len(word2id_ct)+1, 1]#'词':[词序,词频]
            else:
                word2id_ct[word][1] += 1#词频加一

    print("len(word2id_ct):", len(word2id_ct))
    low_freq_words = []
    for word, [word_id, word_freq] in word2id_ct.items():
        if word_freq < min_count:
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id_ct[word]  # 删除低频词

    word2id = {}
    new_id = 1
    for word in word2id_ct.keys():
        word2id[word] = new_id  # word2id = {'<PAD>':0, 'word1':id_1, ......, '<UNK>':id_n}
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    print("len(word2id):", len(word2id))
    return word2id

word2id = vocab_build(texts)

json.dump(word2id, open(mid_path+"word2id.json", 'w',encoding='utf-8'), indent=4)
'''
word2id=json.load(open(mid_path+"word2id.json",encoding='utf-8'))

'''
def google_wordvec_embedding(word2id):
    """
    :param min_count: 最小词频
    :return:  word2id = {'<PAD>':0, 'word1':id_1, ……， '<UNK>':id_n}
    """
    print('reading word embedding data...')
    embedding_matrix = np.random.randn(len(word2id),300)
    from gensim.models import KeyedVectors
    wv_from_text = KeyedVectors.load_word2vec_format(google_word2vec_path, binary=True)
    np.save(mid_path+"Google_word.npy",list(wv_from_text.wv.vocab.keys()))
    count = 0
    unregistered_word = []
    for word in word2id:
        if word in wv_from_text.wv.vocab.keys():
            embedding_matrix[word2id[word]] = wv_from_text[word]
            count += 1
        else:
            unregistered_word.append(word)
    print(len(word2id),len(wv_from_text.vocab),count)
    print(len(unregistered_word))
    np.save(mid_path+"OOV_words.npy",unregistered_word)
    return embedding_matrix


embedding_matrix = google_wordvec_embedding(word2id)
#embedding_matrix = np.random.randn(len(word2id),300)
np.save(mid_path+"embedding_matrix.npy",embedding_matrix)

Google_word = np.load(mid_path+"Google_word.npy")

'''
unregistered_word = np.load(mid_path+"OOV_words.npy")


def to_one_hot(labels, tag_nums):
    length = len(labels)#batch的样本个数
    res = np.zeros((length, tag_nums), dtype=np.float32)
    for i in range(length):
        res[i][labels[i]] = 1.
        #res[i][1] = 1.
    return np.array(res)


x1_max_len = 200
x2_max_len = 200
x3_max_len = 200

def word_to_seq_num(train_data_processed,word2id):
    
    train_seq_num = copy.deepcopy(train_data_processed)
    for index, row in train_data_processed.iterrows():
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ['result','fw1','fw2']:
            output = []
            word_list = []
            for word in row[col_name].split():
                if word not in unregistered_word:
                    word_list.append(word)
            if col_name == "result":
                if len(word_list) != 0:
                    alpha = (1.0 * x1_max_len / len(word_list))
                else:
                    word_list = ['<UNK>'] * x1_max_len
                    alpha = 1.0
                for i in range(x1_max_len):
                    word = word2id[word_list[int(i / alpha)]]
                    output.append(word)
            elif col_name == "fw1":
                if len(word_list) != 0:
                    alpha = (1.0 * x2_max_len / len(word_list))
                else:
                    word_list = ['<UNK>'] * x2_max_len
                    alpha = 1.0
                for i in range(x2_max_len):
                    word = word2id[word_list[int(i / alpha)]]
                    output.append(word)
            else:
                if len(word_list) != 0:
                    alpha = (1.0 * x3_max_len / len(word_list))
                else:
                    word_list = ['<UNK>'] * x3_max_len
                    alpha = 1.0
                for i in range(x3_max_len):
                    word = word2id[word_list[int(i / alpha)]]
                    output.append(word)
            train_seq_num.at[index, col_name] = output
    return train_seq_num

print("data process start:")
train_seq_num = word_to_seq_num(train_data_processed,word2id)
train_data_order = list(range(train_seq_num.shape[0]))
print("train over")
dev_seq_num = word_to_seq_num(dev_data_processed,word2id)
dev_data_order = list(range(dev_seq_num.shape[0]))
print("dev over")
test_seq_num = word_to_seq_num(test_data_processed,word2id)
test_data_order = list(range(test_seq_num.shape[0]))
print("test over")

np.save(mid_path+"train_result_ids.npy",np.array(train_seq_num['result'].tolist())[train_data_order])
np.save(mid_path+"train_fw1_ids.npy",np.array(train_seq_num['fw1'].tolist())[train_data_order])
np.save(mid_path+"train_fw2_ids.npy",np.array(train_seq_num['fw2'].tolist())[train_data_order])
np.save(mid_path+"train_labels.npy",np.array(to_one_hot(np.array(train_seq_num['tt_label'].tolist())-1, 5))[train_data_order])

np.save(mid_path+"dev_result_ids.npy",np.array(dev_seq_num['result'].tolist())[dev_data_order])
np.save(mid_path+"dev_fw1_ids.npy",np.array(dev_seq_num['fw1'].tolist())[dev_data_order])
np.save(mid_path+"dev_fw2_ids.npy",np.array(dev_seq_num['fw2'].tolist())[dev_data_order])
np.save(mid_path+"dev_labels.npy",np.array(to_one_hot(np.array(dev_seq_num['tt_label'].tolist())-1, 5))[dev_data_order])

np.save(mid_path+"test_result_ids.npy",np.array(test_seq_num['result'].tolist())[test_data_order])
np.save(mid_path+"test_fw1_ids.npy",np.array(test_seq_num['fw1'].tolist())[test_data_order])
np.save(mid_path+"test_fw2_ids.npy",np.array(test_seq_num['fw2'].tolist())[test_data_order])
np.save(mid_path+"test_labels.npy",np.array(to_one_hot(np.array(test_seq_num['tt_label'].tolist())-1, 5))[test_data_order])



