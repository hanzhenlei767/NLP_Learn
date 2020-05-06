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
#nltk.download('averaged_perceptron_tagger')

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

'''
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
'''
#处理数据
'''
def preprocess_data(data_all):
	data_processed = pd.DataFrame()
	# 加载停用词
	stopwords_list = stopwords.words('english')

	for index, row in data_all.iterrows():
		for col_name in ['result','fw1','fw2']:
			# 分词+去除停用词
			text = clean_str(row[col_name])
			text_list = nltk.word_tokenize(text)
			filtered = [w for w in text_list if(w not in stopwords_list and w != ' ')]
			pos_tags = nltk.pos_tag(filtered)
			#分词之后的DataFrame
			data_processed.at[index, col_name+"_word"] = " ".join([k[0] for k in pos_tags])
			data_processed.at[index, col_name+"_pos"] = " ".join([k[1] for k in pos_tags])
		data_processed.at[index, "location"] = row["location"]
		data_processed.at[index, "tt_label"] = row["tt_label"]
	return data_processed

'''
'''
构建词表
'''
'''
train_data_processed = preprocess_data(train_data)
dev_data_processed = preprocess_data(dev_data)
test_data_processed = preprocess_data(test_data)

print("train data processed shape:")
print(train_data_processed.shape)
print(train_data_processed.head())
'''
'''
def merge_words(train_all):
	train_data_process = copy.deepcopy(train_all)
	print('merge word process...')
	texts = []
	texts.extend(train_data_process['result_word'].tolist())
	texts.extend(train_data_process['fw1_word'].tolist())
	texts.extend(train_data_process['fw2_word'].tolist())
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
构建词性表
'''
'''
def merge_pos(train_all):
	train_data_process = copy.deepcopy(train_all)
	print('merge pos process...')
	pos = []
	pos.extend(train_data_process['result_pos'].tolist())
	pos.extend(train_data_process['fw1_pos'].tolist())
	pos.extend(train_data_process['fw2_pos'].tolist())
	print(len(pos))
	return pos


train_pos = merge_pos(train_data_processed)
dev_pos = merge_pos(dev_data_processed)
test_pos = merge_pos(test_data_processed)

pos = []
pos.extend(train_pos)
pos.extend(dev_pos)
pos.extend(test_pos)

def pos_build(pos,min_count = -float("inf")):
	"""
	:param min_count: 最小词频
	:return:  word2id = {'<PAD>':0, 'word1':id_1, ……， '<UNK>':id_n}
	"""
	pos2id_ct = {}
	for pos_list in pos:
		for single_pos in pos_list.split():
			if single_pos not in pos2id_ct:
				pos2id_ct[single_pos] = [len(pos2id_ct)+1, 1]#'词':[词序,词频]
			else:
				pos2id_ct[single_pos][1] += 1#词频加一

	print("len(pos2id_ct):", len(pos2id_ct))
	low_freq_pos = []
	for single_pos, [pos_id, pos_freq] in pos2id_ct.items():
		if pos_freq < min_count:
			low_freq_pos.append(single_pos)
	for single_pos in low_freq_pos:
		del pos2id_ct[single_pos]  # 删除低频词

	pos2id = {}
	new_id = 1
	for single_pos in pos2id_ct.keys():
		pos2id[single_pos] = new_id  # pos2id = {'<PAD>':0, 'word1':id_1, ......, '<UNK>':id_n}
		new_id += 1
	pos2id['<UNK>'] = 0
	#pos2id['<PAD>'] = 0
	print("len(pos2id):", len(pos2id))
	return pos2id

pos2id = pos_build(pos)

json.dump(pos2id, open(mid_path+"pos2id.json", 'w',encoding='utf-8'), indent=4)
'''
pos2id=json.load(open(mid_path+"pos2id.json",encoding='utf-8'))



'''
构建idf
'''
'''
result_word = []

result_word.extend(train_data_processed["result_word"].tolist())
result_word.extend(dev_data_processed["result_word"].tolist())
result_word.extend(test_data_processed["result_word"].tolist())

def get_idf(data_all,word2id):
	word2idf = {}
	wordid2idf = {}
	all_keys = list(word2id.keys())[:-2]#最后两个是<PAD>,<UNK>,刨除<PAD>,<UNK>.
	#print(all_keys)
	for i in all_keys:#词汇表
		num = 0
		for k in data_all:#遍历作文
			if i in k.split():
				num += 1
		word2idf[i] = np.log10(float(len(data_all)/(num+1)))
		wordid2idf[word2id[i]] = word2idf[i]
	return word2idf,wordid2idf

word2idf, wordid2idf = get_idf(result_word,word2id)

json.dump(word2idf, open(mid_path+"word2idf.json", 'w',encoding='utf-8'), indent=4)

word2idf=json.load(open(mid_path+"word2idf.json",encoding='utf-8'))



def get_word_idf(data):
	idf = []
	for i in data.split():
		idf.append(word2idf[i])
	idf = " ".join([str(k) for k in idf])
	return idf

train_data_processed["result_idf"] = train_data_processed["result_word"].apply(get_word_idf)
train_data_processed["fw1_idf"] = train_data_processed["fw1_word"].apply(get_word_idf)
train_data_processed["fw2_idf"] = train_data_processed["fw2_word"].apply(get_word_idf)

dev_data_processed["result_idf"] = dev_data_processed["result_word"].apply(get_word_idf)
dev_data_processed["fw1_idf"] = dev_data_processed["fw1_word"].apply(get_word_idf)
dev_data_processed["fw2_idf"] = dev_data_processed["fw2_word"].apply(get_word_idf)

test_data_processed["result_idf"] = test_data_processed["result_word"].apply(get_word_idf)
test_data_processed["fw1_idf"] = test_data_processed["fw1_word"].apply(get_word_idf)
test_data_processed["fw2_idf"] = test_data_processed["fw2_word"].apply(get_word_idf)


print("train data processed idf shape:")
print(train_data_processed.shape)
print(train_data_processed.head())


train_data_processed.to_csv(mid_path+"train_data_processed.csv", index=False)
dev_data_processed.to_csv(mid_path+"dev_data_processed.csv", index=False)
test_data_processed.to_csv(mid_path+"test_data_processed.csv", index=False)
'''
train_data_processed = pd.read_csv(mid_path+"train_data_processed.csv", \
	header=0, names=['result_word','result_pos','fw1_word','fw1_pos', \
	'fw2_word', 'fw2_pos','location','tt_label', 'result_idf', 'fw1_idf', 'fw2_idf'])

dev_data_processed = pd.read_csv(mid_path+"dev_data_processed.csv", \
	header=0, names=['result_word','result_pos','fw1_word','fw1_pos', \
	'fw2_word', 'fw2_pos','location','tt_label', 'result_idf', 'fw1_idf', 'fw2_idf'])

test_data_processed = pd.read_csv(mid_path+"test_data_processed.csv", \
	header=0, names=['result_word','result_pos','fw1_word','fw1_pos', \
	'fw2_word', 'fw2_pos','location','tt_label', 'result_idf', 'fw1_idf', 'fw2_idf'])


print("train data processed idf shape:")
print(train_data_processed.shape)
print(train_data_processed.head())


'''
构建预训练嵌入词向量
'''
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


'''
one-hot标签
'''
def to_one_hot(labels, tag_nums):
	length = len(labels)#batch的样本个数
	res = np.zeros((length, tag_nums), dtype=np.float32)
	for i in range(length):
		res[i][int(labels[i])] = 1.
		#res[i][1] = 1.
	return np.array(res)

x1_max_len = 150
x2_max_len = 150
x3_max_len = 150

def word_to_seq_num(train_data_processed,word2id):
	
	train_seq_num = copy.deepcopy(train_data_processed)
	yw_word_all = []
	yw_pos_all = []
	yw_idf_all = []
	yw_position_all = []

	fw1_word_all = []
	fw1_pos_all = []
	fw1_idf_all = []
	fw1_position_all = []

	fw2_word_all = []
	fw2_pos_all = []
	fw2_idf_all = []
	fw2_position_all = []
	for index, row in train_data_processed.iterrows():
		# 分别遍历每行的两个句子，并进行分词处理
		for col_name in ['result_word','fw1_word','fw2_word']:
			tmp_word_list = []
			tmp_pos_list = []
			tmp_idf_list = []
			tmp_position_list = []

			word_list = str(row[col_name]).split()
			pos_list = str(row[col_name[:-4]+'pos']).split()
			idf_list = str(row[col_name[:-4]+'idf']).split()
			for index in range(len(word_list)):
				if word_list[index] not in unregistered_word:
					tmp_word_list.append(word_list[index])
					tmp_pos_list.append(pos_list[index])
					tmp_idf_list.append(idf_list[index])
			tmp_position_list = [i+1 for i in range(len(tmp_word_list))]

			if col_name == "result_word":
				if len(tmp_word_list) != 0:
					alpha = (1.0 * x1_max_len / len(tmp_word_list))
				else:
					tmp_word_list = ['<UNK>'] * x1_max_len
					tmp_pos_list = ['<UNK>'] * x1_max_len
					tmp_idf_list = [0] * x1_max_len
					tmp_position_list = [0] * x1_max_len
					alpha = 1.0
				
				yw_word_all.append([word2id[tmp_word_list[int(i / alpha)]] for i in range(x1_max_len)])
				yw_pos_all.append([pos2id[tmp_pos_list[int(i / alpha)]] for i in range(x1_max_len)])
				yw_idf_all.append([tmp_idf_list[int(i / alpha)] for i in range(x1_max_len)])
				yw_position_all.append([tmp_position_list[int(i / alpha)] for i in range(x1_max_len)])

			elif col_name == "fw1_word":
				if len(tmp_word_list) != 0:
					alpha = (1.0 * x2_max_len / len(tmp_word_list))
				else:
					tmp_word_list = ['<UNK>'] * x2_max_len
					tmp_pos_list = ['<UNK>'] * x2_max_len
					tmp_idf_list = [0] * x2_max_len
					tmp_position_list = [0] * x2_max_len
					alpha = 1.0
				
				fw1_word_all.append([word2id[tmp_word_list[int(i / alpha)]] for i in range(x2_max_len)])
				fw1_pos_all.append([pos2id[tmp_pos_list[int(i / alpha)]] for i in range(x2_max_len)])
				fw1_idf_all.append([tmp_idf_list[int(i / alpha)] for i in range(x2_max_len)])
				fw1_position_all.append([tmp_position_list[int(i / alpha)] for i in range(x2_max_len)])
			else:
				if len(tmp_word_list) != 0:
					alpha = (1.0 * x3_max_len / len(tmp_word_list))
				else:
					tmp_word_list = ['<UNK>'] * x3_max_len
					tmp_pos_list = ['<UNK>'] * x3_max_len
					tmp_idf_list = [0] * x3_max_len
					tmp_position_list = [0] * x3_max_len
					alpha = 1.0
				
				fw2_word_all.append([word2id[tmp_word_list[int(i / alpha)]] for i in range(x3_max_len)])
				fw2_pos_all.append([pos2id[tmp_pos_list[int(i / alpha)]] for i in range(x3_max_len)])
				fw2_idf_all.append([tmp_idf_list[int(i / alpha)] for i in range(x3_max_len)])
				fw2_position_all.append([tmp_position_list[int(i / alpha)] for i in range(x3_max_len)])

	return yw_word_all,yw_pos_all,yw_idf_all,yw_position_all, \
		   fw1_word_all,fw1_pos_all,fw1_idf_all,fw1_position_all, \
		   fw2_word_all,fw2_pos_all,fw2_idf_all,fw2_position_all

print("data process start:")
train_seq = word_to_seq_num(train_data_processed,word2id)
print("train over")
dev_seq = word_to_seq_num(dev_data_processed,word2id)
print("dev over")
test_seq = word_to_seq_num(test_data_processed,word2id)
print("test over")

np.save(mid_path+"train_yw_word.npy",np.array(train_seq[0]))
np.save(mid_path+"train_yw_pos.npy",np.array(train_seq[1]))
np.save(mid_path+"train_yw_idf.npy",np.array(train_seq[2]))
np.save(mid_path+"train_yw_position.npy",np.array(train_seq[3]))

np.save(mid_path+"train_fw1_word.npy",np.array(train_seq[4]))
np.save(mid_path+"train_fw1_pos.npy",np.array(train_seq[5]))
np.save(mid_path+"train_fw1_idf.npy",np.array(train_seq[6]))
np.save(mid_path+"train_fw1_position.npy",np.array(train_seq[7]))

np.save(mid_path+"train_fw2_word.npy",np.array(train_seq[8]))
np.save(mid_path+"train_fw2_pos.npy",np.array(train_seq[9]))
np.save(mid_path+"train_fw2_idf.npy",np.array(train_seq[10]))
np.save(mid_path+"train_fw2_position.npy",np.array(train_seq[11]))
np.save(mid_path+"train_labels.npy",np.array(to_one_hot(np.array(train_data_processed['tt_label'].tolist())-1, 5)))


np.save(mid_path+"dev_yw_word.npy",np.array(dev_seq[0]))
np.save(mid_path+"dev_yw_pos.npy",np.array(dev_seq[1]))
np.save(mid_path+"dev_yw_idf.npy",np.array(dev_seq[2]))
np.save(mid_path+"dev_yw_position.npy",np.array(dev_seq[3]))

np.save(mid_path+"dev_fw1_word.npy",np.array(dev_seq[4]))
np.save(mid_path+"dev_fw1_pos.npy",np.array(dev_seq[5]))
np.save(mid_path+"dev_fw1_idf.npy",np.array(dev_seq[6]))
np.save(mid_path+"dev_fw1_position.npy",np.array(dev_seq[7]))

np.save(mid_path+"dev_fw2_word.npy",np.array(dev_seq[8]))
np.save(mid_path+"dev_fw2_pos.npy",np.array(dev_seq[9]))
np.save(mid_path+"dev_fw2_idf.npy",np.array(dev_seq[10]))
np.save(mid_path+"dev_fw2_position.npy",np.array(dev_seq[11]))
np.save(mid_path+"dev_labels.npy",np.array(to_one_hot(np.array(dev_data_processed['tt_label'].tolist())-1, 5)))

np.save(mid_path+"test_yw_word.npy",np.array(test_seq[0]))
np.save(mid_path+"test_yw_pos.npy",np.array(test_seq[1]))
np.save(mid_path+"test_yw_idf.npy",np.array(test_seq[2]))
np.save(mid_path+"test_yw_position.npy",np.array(test_seq[3]))

np.save(mid_path+"test_fw1_word.npy",np.array(test_seq[4]))
np.save(mid_path+"test_fw1_pos.npy",np.array(test_seq[5]))
np.save(mid_path+"test_fw1_idf.npy",np.array(test_seq[6]))
np.save(mid_path+"test_fw1_position.npy",np.array(test_seq[7]))

np.save(mid_path+"test_fw2_word.npy",np.array(test_seq[8]))
np.save(mid_path+"test_fw2_pos.npy",np.array(test_seq[9]))
np.save(mid_path+"test_fw2_idf.npy",np.array(test_seq[10]))
np.save(mid_path+"test_fw2_position.npy",np.array(test_seq[11]))
np.save(mid_path+"test_labels.npy",np.array(to_one_hot(np.array(test_data_processed['tt_label'].tolist())-1, 5)))

