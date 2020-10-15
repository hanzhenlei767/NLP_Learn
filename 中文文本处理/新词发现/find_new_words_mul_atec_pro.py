import os
import codecs

COMMON_DIC_PATH = 'common.dic' #词典目录

def data_reader(filename, cn_only=False):
    '''
    1.打开读取文件
    2.cn_only：过滤非中文字符
    '''
    try:
        with codecs.open(filename, encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with codecs.open(filename, encoding='gbk') as f:
            text = f.read()
    if cn_only:
        # 过滤非中文字符.
        re_non_cn = re.compile('[^\u4e00-\u9fa5]+')
        text = re_non_cn.sub('', text)
    return text

class Dictionary:
    """
    一个加载/更新词典文件的类: common.dic.
    new words = all extraction words - common words 
    为了提升新词发现的结果，需要用特定的语料更新common.dic.
    """
    def __init__(self, common_dic_path=None):
        if common_dic_path:
            self._common_dic_path = common_dic_path
            print("使用用户的公共词典目录 `{}`".format(self._common_dic_path))
        else:
            self._common_dic_path = COMMON_DIC_PATH
            print("使用默认的公共词典目录 `{}`".format(self._common_dic_path))

        self.dictionary = set()
        self._init()

    def _init(self):
        """初始化公共词典."""
        common_dic = data_reader(self._common_dic_path)
        vocab = common_dic.strip().split('\n')
        for word in vocab:
            if word:#词非空
                self.dictionary.add(word)
        print("从 `{}` 文件中初始化了`{}`个公共词".format(self._common_dic_path,len(vocab)))

    def __contains__(self, item):
        '''
        魔术函数：对象支持in操作
        '''
        return item in self.dictionary

    def __iter__(self):
        """
        魔术函数：对象支持迭代
        """
        for word in self.dictionary:
            yield word

    def __len__(self):
        """
        魔术函数：对象支持len方法
        """
        return len(self.dictionary)

    def load(self):
        """
        返回词典
        """
        return self.dictionary

    def add(self, vocab):
        """
        添加词到公共词典,并且更新公共词典
        vocab：词列表
        """
        with codecs.open(self._common_dic_path, 'a+', encoding='UTF-8') as fo:
            for word in vocab:
                if word not in self.dictionary:
                    self.dictionary.add(word)
                    fo.write('\n'+word)
                    #print("添加 `{}` 到文件 `{}`.".format(word, self._common_dic_path))

    def add_from_file(self, vocab_file):
        """
        添加词到公共词典,并且更新公共词典
        vocab_file：词典文件
        """
        user_dic = data_reader(vocab_file)
        vocab = user_dic.strip().split('\n')
        with codecs.open(self._common_dic_path, 'a+', encoding='UTF-8') as fo:
            for word in vocab:
                if word not in self.dictionary:
                    self.dictionary.add(word)
                    fo.write('\n'+word)
                    print("添加 `{}` 到文件 `{}`.".format(word, self._common_dic_path))

    def remove(self, vocab):
        """
        删除公共词典的词，并且更新公共词典
        vocab：词列表
        """
        for word in vocab:
            if word in self.dictionary:
                self.dictionary.remove(word)
                print("删除 `{}` 到文件 `{}`.".format(word, self._common_dic_path))
        with codecs.open(self._common_dic_path, 'w', encoding='UTF-8') as fo:
            for word in self.dictionary:
                fo.write(word + '\n')

    def remove_from_file(self, vocab_file):
        """
        删除公共词典的词，并且更新公共词典
        vocab_file：词列表文件
        """
        user_dic = data_reader(vocab_file)
        vocab = user_dic.strip().split('\n')
        for word in vocab:
            if word in self.dictionary:
                self.dictionary.remove(word)
                print("删除 `{}` 到文件 `{}`.".format(word, self._common_dic_path))
        with codecs.open(self._common_dic_path, 'w', encoding='UTF-8') as fo:
            for word in self.dictionary:
                fo.write(word + '\n')


FILE_PATH = "./token_freq_pos%40350k_jieba.txt"
#读取jieba词表
def construct_dict( file_path ):

    word_freq = {}
    with open(file_path, "r",encoding='UTF-8') as f:
        for line in f:
            info = line.split()
            word = info[0]
            frequency = info[1]
            word_freq[word] = frequency
    return word_freq
phrase_freq = construct_dict( FILE_PATH )


common_dic = Dictionary()
common_dic.add(phrase_freq.keys())


common_dic = Dictionary()

from collections import defaultdict
import re


class TextUtils:
    @staticmethod
    def is_chinese(char):
        return u'\u4e00' <= char <= u'\u9fa5'

    @staticmethod
    def is_english(char):
        return char.isalpha()

    @staticmethod
    def is_numeric(char):
        return char.isdigit()

    @staticmethod
    def match(src, off, dest):
        '''
        src:文本
        off:出现dest[0]的位置
        dest:匹配的词
        '''
        src_len = len(src)
        dest_len = len(dest)
        for i in range(dest_len):
            if src_len <= off + i:
                return False
            if src[off+i] != dest[i]:
                return False
        return True
class CnTextIndexer:

    def __init__(self, document):
        self._document = document
        self._doc_len = len(document)
        self._char_pos_map = defaultdict(list)#字典{key:[v1,v2,...]}
        self._char_cnt_map = defaultdict(int)#字典{key:v1}
        self._init_char_mapping()

    def _init_char_mapping(self):
        """初始化_char_pos_map和_char_cnt_map."""
        for ix, char in enumerate(self._document):
            self._char_pos_map[char].append(ix)
            self._char_cnt_map[char] += 1
        print("用document初始化char pos和char count map..")

    def count(self, target):
        """Target text count in document"""
        if not target:
            return 0

        index = self._char_pos_map[target[0]]
        if len(index) == 0:#不存在该字
            return 0

        if len(target) == 1:#单字
            return len(index)

        count = 0
        for pos in index:
            if TextUtils.match(self._document, pos, target):
                count += 1

        return count

    def find(self, target):
        """All target text matching start index in document
        Yields:
            index_list
        """
        if not target:
            yield 0
        # 获取候选词的第一个字符出现在文本中的所有下标
        index = self._char_pos_map[target[0]]
        if len(index) == 0:
            yield 0

        if len(target) == 1:
            for pos in index:
                yield pos

        for pos in index:
            if TextUtils.match(self._document, pos, target):
                yield pos
    def __getitem__(self, index):
        '''
        魔术函数：对象通过下标可以直接访问文档中的字
        '''
        if index < 0 or index > self._doc_len-1:
            return ""
        return self._document[index]
    @property
    def char_cnt_map(self):
        return self._char_cnt_map


class CnTextSelector:

    def __init__(self, document, min_len=2, max_len=5):
        """
        Args:
            document: String, filtered chinese corpus.
            min_len: candidate word min length.
            max_len: candidate word max length.
        """
        self._document = document
        self._max_len = max_len
        self._min_len = min_len
        self._doc_len = len(document)

    def generate(self):
        """Returns:
            A generator of candidate chinese word from document.
        """
        for pos in range(self._doc_len-self._min_len):
            for cur_len in range(self._min_len, self._max_len+1):
                yield pos, self._document[pos:pos+cur_len]


import math



class WordCountDict(dict):
    def add(self, word):
        self[word] = self.get(word) + 1

    def get(self, word):  # override dict.get method
        return super(WordCountDict, self).get(word, 0)

    def count(self):
        return sum(self.values())


class EntropyJudger:
    """利用熵和实值率来判断一个候选词是否为中文词。"""

    def __init__(self, document, least_cnt_threshold=5, solid_rate_threshold=0.018, entropy_threshold=1.92):
        """
        Args:
            least_cnt_threshold: 一个字最少出现的次数，如果小于这个值则不能通过
            solid_rate_threshold: p(candidate)/p(candidate[0]) * p(candidate)/p(candidate[1]) * ...
            entropy_threshold: min(left_char_entropy, right_char_entropy), The smaller this values is, 
                more new words you will get, but with less accuracy.
        """
        self._least_cnt_threshold = least_cnt_threshold#最少出现的次数
        self._solid_rate_threshold = solid_rate_threshold#实值率
        self._entropy_threshold = entropy_threshold#熵-熵值越大不确定性越高
        self._indexer = CnTextIndexer(document)

    def judge(self, candidate):
        '''
        如果实值率或熵小于阈值则判别为不是新词
        '''
        solid_rate = self._get_solid_rate(candidate)#实值率
        entropy = self._get_entropy(candidate)#熵
        if solid_rate < self._solid_rate_threshold or entropy < self._entropy_threshold:
            return False
        return True

    def _get_solid_rate(self, candidate):
        '''
        实值率
        '''
        if len(candidate) < 2:
            return 1.0
        cnt = self._indexer.count(candidate)  # 候选词的数量
        if cnt < self._least_cnt_threshold:  # 候选词的数量小于最小词数的阈值
            return 0.0
        rate = 1.0
        for c in candidate:
            rate *= cnt / self._indexer.char_cnt_map[c]  # 在文档里候选字计数
        return math.pow(rate, 1/float(len(candidate))) * math.sqrt(len(candidate))  # interesting

    def _get_entropy(self, candidate):
        '''
        熵
        '''
        left_char_dic = WordCountDict()
        right_char_dic = WordCountDict()
        candidate_pos_generator = self._indexer.find(candidate)
        for pos in candidate_pos_generator:
            c = self._indexer[pos-1]
            left_char_dic.add(c)
            c = self._indexer[pos+len(candidate)]
            right_char_dic.add(c)

        previous_total_char_cnt = left_char_dic.count()
        next_total_char_cnt = right_char_dic.count()
        previous_entropy = 0.0
        next_entropy = 0.0
        for char, count in left_char_dic.items():  # efficient
            prob = count / previous_total_char_cnt
            previous_entropy -= prob * math.log(prob)
        for char, count in right_char_dic.items():
            prob = count / next_total_char_cnt
            next_entropy -= prob * math.log(prob)
        return min(previous_entropy, next_entropy)  # 返回前后信息熵中较小的一个
import time
import codecs
import threading    # 引用线程模块
import multiprocessing   # 导入进程模块
'''
class WordExtractMul:

    def __init__(self,
                 corpus_file,
                 common_words_file=None,
                 min_candidate_len=2,
                 max_candidate_len=5,
                 least_cnt_threshold=5,
                 solid_rate_threshold=0.018,
                 entropy_threshold=1.92,
                 all_words=False):
        if not corpus_file:
            raise ValueError("Corpus file is empty, please specify corpus file path.")
        self._document = data_reader(corpus_file, cn_only=True)
        self._common_dic = common_words_file
        self._min_candidate_len = min_candidate_len
        self._max_candidate_len = max_candidate_len
        self._least_cnt_threshold = least_cnt_threshold
        self._solid_rate_threshold = solid_rate_threshold
        self._entropy_threshold = entropy_threshold
        self._all_words = all_words
        if not self._all_words:
            self.dictionary = Dictionary(self._common_dic)
        else:
            print("Extract all words mode, if you only want new words, set new_words=False to new words mode.")

    stopwords = {'我', '你', '您', '他', '她', '谁', '哪', '那', '这',
                 '的', '了', '着', '也', '是', '有', '不', '在', '与',
                 '呢', '啊', '呀', '吧', '嗯', '哦', '哈', '呐'}

    def extract(self):
        """New word discover is based on statistic and entropy, better to sure
        document size is in 100kb level, or you may get a unsatisfied result.
        """
        
        length = len(self._document)
        selector = CnTextSelector(self._document, self._min_candidate_len, self._max_candidate_len)
        judger = EntropyJudger(self._document, self._least_cnt_threshold, self._solid_rate_threshold, self._entropy_threshold)
        candidate_generator = selector.generate()
        print("Document Length: {}".format(length))
        indexer = CnTextIndexer(self._document)
        def process_thread(i,candidate_list):
            s_time = time.time()
            new_word_set = set()
            print(len(candidate_list))
            for pos, candidate in candidate_list:#遍历候选词
                # 关于怎么用停用词
                if candidate[0] in self.stopwords or candidate[-1] in self.stopwords:
                    #判断候选词的开头和结尾是否是停用词
                    continue
                if candidate in new_word_set:
                    continue
                if not self._all_words:
                    if candidate in self.dictionary:
                        continue
                if judger.judge(candidate):
                    new_word_set.add(candidate)
#                 if (pos+1) % 10000 == 0 and pos != pos_tmp:
#                     pos_tmp = pos
#                     print("Process {}/{} characters in document".format(pos+1, len(candidate_list)))
            time_elapse = time.time() - s_time
            print("Process:{},Time Cost: {}s".format(i,time_elapse))
            #print(self.new_word_set)
            with codecs.open("result/"+"new_word_"+str(i), 'w', encoding='utf-8') as fo:
                for w in new_word_set:
                    word_cnt = indexer.count(w)
                    fo.write(w + '\t' + str(word_cnt) + '\n')

        all_candidate = list(candidate_generator)

        
        for i in range(len(all_candidate)//80000+1):
            handle_thread = multiprocessing.Process(target=process_thread,args=(i,all_candidate[80000*i:80000*(i+1)],))
            handle_thread.start()#启动线程 
            #handle_thread.join()#等待第一个进程执行完
            print(i)
'''
import time
import codecs
import threading    # 引用线程模块
import multiprocessing   # 导入进程模块
class WordExtractMul:

    def __init__(self,
                 corpus_file,
                 common_words_file=None,
                 min_candidate_len=2,
                 max_candidate_len=5,
                 least_cnt_threshold=5,
                 solid_rate_threshold=0.018,
                 entropy_threshold=1.92,
                 all_words=False):
        if not corpus_file:
            raise ValueError("Corpus file is empty, please specify corpus file path.")
        self._document = data_reader(corpus_file, cn_only=True)
        self._common_dic = common_words_file
        self._min_candidate_len = min_candidate_len
        self._max_candidate_len = max_candidate_len
        self._least_cnt_threshold = least_cnt_threshold
        self._solid_rate_threshold = solid_rate_threshold
        self._entropy_threshold = entropy_threshold
        self._all_words = all_words
        if not self._all_words:
            self.dictionary = Dictionary(self._common_dic)
        else:
            print("Extract all words mode, if you only want new words, set new_words=False to new words mode.")

    stopwords = {'我', '你', '您', '他', '她', '谁', '哪', '那', '这',
                 '的', '了', '着', '也', '是', '有', '不', '在', '与',
                 '呢', '啊', '呀', '吧', '嗯', '哦', '哈', '呐'}

    def extract(self):
        """New word discover is based on statistic and entropy, better to sure
        document size is in 100kb level, or you may get a unsatisfied result.
        """
        
        length = len(self._document)
        selector = CnTextSelector(self._document, self._min_candidate_len, self._max_candidate_len)
        judger = EntropyJudger(self._document, self._least_cnt_threshold, self._solid_rate_threshold, self._entropy_threshold)
        candidate_generator = selector.generate()
        print("Document Length: {}".format(length))
        indexer = CnTextIndexer(self._document)
        def process_thread(i,candidate_list):
            s_time = time.time()
            new_word_set = set()
            print(len(candidate_list))
            for candidate in candidate_list:#遍历候选词
                # 关于怎么用停用词
                if candidate[0] in self.stopwords or candidate[-1] in self.stopwords:
                    #判断候选词的开头和结尾是否是停用词
                    continue
                if candidate in new_word_set:
                    continue
                if not self._all_words:
                    if candidate in self.dictionary:
                        continue
                if judger.judge(candidate):
                    new_word_set.add(candidate)
            time_elapse = time.time() - s_time
            print("Process:{},Time Cost: {}s".format(i,time_elapse))
            #print(self.new_word_set)
            with codecs.open("result/"+"new_word_"+str(i), 'w', encoding='utf-8') as fo:
                for w in new_word_set:
                    word_cnt = indexer.count(w)
                    fo.write(w + '\t' + str(word_cnt) + '\n')

                    
        all_candidate = list(candidate_generator)
        
        prefix_dictionary = set()
        for i in all_candidate:
            prefix_dictionary.add(i[1])
        
        print(len(prefix_dictionary))
        
        for i in range(len(prefix_dictionary)//20000+1):
            handle_thread = multiprocessing.Process(target=process_thread,args=(i,list(prefix_dictionary)[20000*i:20000*(i+1)],))
            handle_thread.start()#启动线程 
            #handle_thread.join()#等待第一个进程执行完
            print(i)
new_word_finder = WordExtractMul('atec_nlp_sim_train.csv')
new_word_finder.extract()