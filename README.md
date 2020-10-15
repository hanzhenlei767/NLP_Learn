# NLP_Learn
# 学习笔记整理-前沿paper跟踪翻译复现：
## 1.NLP基础
* 1).环境搭建
* 2).数据集下载探索
* 3).文本表示
* 4).分词，词频统计，抽取特征
* 5).传统机器学习--朴素贝叶斯
* 6).传统机器学习--SVM 
* 7).传统机器学习--LDA
* 8).神经网络基础
* 9).简单神经网络
* 10).卷积神经网络基础

## 2.NLP预训练模型
* 1).word2vec/doc2vec
  
  Distributed Representations of Words and Phrases and their Compositionality
  
  单词和短语的分布式表示及其组合性
  
  Efficient Estimation of Word Representations in Vector Space
  
  向量空间中字表示的有效估计
  
  Distributed Representations of Sentences and Documents
  
  分发句子和文档的表示形式
  
  Enriching Word Vectors with Subword Information
  
  用Subword信息丰富词向量
  
  Fair is Better than Sensational Man is to Doctor as Woman is to Doctor
  
  男人是医生，正如女人是医生一样
  
  Linguistic Regularities in Continuous Space Word Representations
  
  连续空间词表征中的语言规律
  
* 2).BERT
  
  Attention Is All You Need
  
  你需要的只是Attention
  
  BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
  
  BER论文翻译
  
  BERT用于文本相似度计算源码
  
  BERT用于文本相似度计算的多GPU版本源码
  
* 3).ELMo
  
  ELMo_Deep contextualized word representations
  
* 4).ERNIE

  ERNIE: Enhanced Language Representation with Informative Entities

  ERNIE:用信息实体增强语言表示

* 5).RoBERTa

  RoBERTa: A Robustly Optimized BERT Pretraining Approach

* 6).ALBERT

  ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS
  
  ALBERT论文翻译
  
  ALBERT用于文本相似度计算源码
  
  ALBERT用于文本相似度计算的多GPU版本源码

* 7).XLNet

  Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

  XLNet: Generalized Autoregressive Pretraining for Language Understanding

  XLNet：广义自回归预训练语言模型

* 8).T5

  Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

* 9).SHA-RNN

  Single Headed Attention RNN: Stop Thinking With Your Head

  单头注意力RNN:停止用你的头思考
  
* 10).DistilBERT

  DistilBERT, a distilled version of BERT: smaller,faster, cheaper and lighter

  蒸馏BERT，BERT的蒸馏版:更小，更快，更便宜，更轻

* 11).FastBERT

  FastBERT:a Self-distilling BERT with Adaptive Inference Time

  FastBERT:一个自蒸馏的BERT与自适应推理时间
  
* 12).TinyBERT

  TINYBERT: DISTILLING BERT FOR NATURAL LAN GUAGE UNDERSTANDING

  TINYBERT，自然语言理解的蒸馏BERT
  
* 13).GloVe

  GloVe: Global Vectors for Word Representation

  GloVe:用于单词表示的全局向量
  
* 14).Synthesizer

  SYNTHESIZER:Rethinking Self-Attention in Transformer Models

  SYNTHESIZER:重新思考Transformer模型中的Self-Attention 

## 3.中文文本处理
* 1).分词
* 2).错词检查
* 3).新词发现


## 4.对话系统
* 1).LSTM-based Deep Learning Models for Non-factoid Answer Selection
  
  基于LSTM的非事实性答案选择深度学习模型
  
* 2).Denoising Distantly Supervised Open-Domain Question Answering
  
  去噪远距离监督开放域问题的回答
  
* 3).Reading Wikipedia to Answer Open-Domain Questions
  
  阅读维基百科来回答开放领域的问题

## 5.文本匹配
* 1).MIX：Multi-Channel Information Crossing for Text Matching
  
  MIX:多通道信息交叉用于文本匹配
  
* 2).A COMPARE-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES
  
  用于匹配文本序列的比较-聚合模型
  
* 3).MatchPyramid：Text Matching as Image Recognition
  
  文本匹配作为图像识别
  
* 4).A Decomposable Attention Model for Natural Language Inference

* 5).Convolutional Neural Network Architectures for Matching Natural Language Sentences

* 6).DRMM:A Deep Relevance Matching Model for Ad-hoc Retrieval

* 7).A Tensor Neural Network with Layerwise Pretraining_ Towards Effective Answer Retrieval

* 8).CNTN:Convolutional Neural Tensor Network Architecture for Community-based Question Answering

* 9).RE2:Simple and Effective Text Matching with Richer Alignment Features

* 10).WMD:From Word Embeddings To Document Distances


## 6.对抗训练
* 1).FREELB: 

  FREELB: ENHANCED ADVERSARIAL TRAINING FOR LANGUAGE UNDERSTANDING

  Free Large-Batch：增强对抗性训练的语言理解

## 7.知识图谱
### 实体识别：
* 1).Chinese NER Using Lattice LSTM
  
  使用网格LSTM的中文命名实体识别
  
* 2).Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning
  
  使用正未标记学习的远程监督命名实体识别
  
* 3).Fine-Grained Entity Typing in Hyperbolic Space
  
  在双曲空间中细粒度实体类型关系抽取：

### 关系抽取：
* 1).Enriching Pre-trained Language Model with Entity Information for Relation Classification
  
  用实体信息丰富预训练的语言模型进行关系分类-不适用远程监督数据
  
* 2).Relation Classification via Convolutional Deep Neural Network
  
  基于卷积深度神经网络的关系分类
  
* 3).Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
  
  基于注意力的双向LSTM关系分类
  
* 4).Neural Relation Extraction with Selective Attention over Instances
  
  具有实例选择性注意力的神经关系抽取
  
* 5).Relation Classification via Multi-Level Attention CNN 
  
  利用多层注意力CNNs进行关系分类
  
* 6).A Survey of Deep Learning Methods for Relation Extraction

* 7).Convolutional Sequence to Sequence Learning


### 实体关系联合抽取：
* 1).Entity-Relation Extraction as Multi-turn Question Answering
  
  多轮QA用于实体关系抽取


## 8.知识蒸馏
* 1).Distilling-the-knowledge-in-neural-network

  从神经网络中提取知识
  
  
## 9.模型框架
* 1).LightGBM: A Highly Efficient Gradient Boosting Decision Tree

  LightGBM:一个高效的梯度增强决策树
  
* 2).XGBoost: A Scalable Tree Boosting System

  XGBoost:一个可伸缩的树增强系统
  
## 10.深度学习知识点
* 1).Dropout: A Simple Way to Prevent Neural Networks from Overfitting

  Dropout:防止神经网络过拟合的简单方法
  
* 2).Improving Neural Networks with Dropout

  用Dropout改进神经网络
  
* 3).Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

  Batch Normalization:通过减少内部协变量移位加速深度网络训练
  
* 4).Layer Normalization

  Layer Normalization
  
* 5).Instance Normalization: The Missing Ingredient for Fast Stylization

  Instance Normalization::快速格式化所缺少的元素
  
* 6).Sequence to Sequence Learning with Neural Networks

  利用神经网络进行序列到序列学习
  
* 7).Focal Loss for Dense Object Detection

  分类任务中类别加权，难分易分样本加权
  
* 8).Dice Loss for Data-imbalanced NLP Tasks

  适用于分类任务中评价指标是F1,且样本类别不均衡
  
* 9).Rethinking the Inception Architecture for Computer Vision

  label smoothing:分类任务中降低标注错误数据带来的影响
  
* 10).Huber Loss:回归任务中L1和L2的结合，降低标注错误数据的影响
  
* 11).Quantile Loss:回归任务中把L1中的正负损失加权