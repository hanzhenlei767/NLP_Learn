# Entity-Relation Extraction as Multi-turn Question Answering

The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). 

**Entity-Relation Extraction as Multi-turn Question Answering (ACL 2019)**<br>
Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan, Duo Chai, Mingxin Zhou and Jiwei Li <br> 
Accepted by [ACL 2019](https://arxiv.org/pdf/1905.05529.pdf) <br>
If you find this repo helpful, please cite the following:
```text
@inproceedings{li-etal-2019-entity,
    title = "Entity-Relation Extraction as Multi-Turn Question Answering",
    author = "Li, Xiaoya  and
      Yin, Fan  and
      Sun, Zijun  and
      Li, Xiayu  and
      Yuan, Arianna  and
      Chai, Duo  and
      Zhou, Mingxin  and
      Li, Jiwei",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1129",
    doi = "10.18653/v1/P19-1129",
    pages = "1340--1350"
}
```
 

## Overview

In this paper, we regard joint entity-relation extraction as a **multi-turn question answering task (multi-turn QA)**: each kind of entity and relation can be described by a QA template, through which the corresponding entity or relation can be extracted from raw texts as answers. s<br> 

For example, 

In addition to multi-QA, we also utilize **reinforcement learning** to better extract answers with long template chains. We also design a strategy to automatically generate question templates and answers. More details please refer to our paper.


## Contents
1. [Experimental Results](#experimental-results)
2. [Data Preparation](#data-preparation) 
2. [Dependencies](#dependencies)
3. [Usage](#usage)
4. [FAQ](#faq)


## Experimental Results

Evalutations are conducted on the widely used datasets `ACE 2004`, `ACE 2005` and `CoNLL 2004`.  
We report micro precision, recall and F1-score for entity and relation extractions. 
We only list the experimental comparion between the proposed method and **previous** `state-of-the-art` model. More experimental comparions are shown in paper. 

- Results on **ACE 2004**:

   *Models* | Enity P | Entity R | Entity F | Relation P | Relation R | Relation F
   --- | --- | --- | --- | --- | --- | --- 
   Miwa and Bansal (2016) | 80.8 | 82.9 | 81.8 | 48.7 | 48.1 | 48.4 
  Multi-turn QA| 84.4 | 82.9 | **83.6** | 50.1 | **48.7** | **49.4 (+1.0)** 
  
- Results on **ACE 2005**:

  | *Models* | Enity P | Entity R | Entity F | Relation P | Relation R | Relation F|
  | --- | --- | --- | --- | --- | --- | --- |
  |Sun et al. (2018) |83.9 s|83.2| 83.6| 64.9| 55.1| 59.6|
  |Multi-turn QA |84.7 |84.9|**84.8** |64.8| 56.2| **60.2 (+0.6)**|
  
- Results on **CoNLL 2004**:

  | *Models* | Enity P | Entity R | Entity F | Relation P | Relation R | Relation F|
  | --- | --- | --- | --- | --- | --- | --- |
  |Zhang et al. (2017) |– |–| 85.6 |– |–| 67.8|
  |Multi-turn QA | 89.0 | 86.6 | **87.8** | 69.2 | 68.2 | **68.9 (+1.1)**|


## Data Preparation

We will release preprocessed and source data files for our experiments. 
You can download the preprocessed datasets and source data files from [Google  Drive](./docs/data_download.md). 

For data processing, you can follow the [guidance](./docs/data_preprocess.md) to generate your own QA-based relation extraction files.

    
## Dependencies 

* Package dependencies: 
```bash 
python >= 3.6
PyTorch == 1.1.0
pytorch-pretrained-bert == 0.6.1 
```
* Download and unzip `BERT-Large, English` pretrained model. Follow the following command to convert the model's checkpoint from Tensorflow to PyTorch. 
    
    ```bash 
    export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12

    pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
    $BERT_BASE_DIR/bert_model.ckpt \
    $BERT_BASE_DIR/bert_config.json \
    $BERT_BASE_DIR/pytorch_model.bin
    ```


## Usage
As an example, the following command trains the proposed mothod on ACE 2005. 

```bash 
base_path=/home/work/Entity-Relation-As-Multi-Turn-QA
bert_model=/data/BERT_BASE_DIR/cased_L-24_H-1024_A-16
data_dir=/data/ace05-qa

config_path=Entity-Relation-As-Multi-Turn-QA/configs/eng_large_case_bert.json
task_name=ner
max_seq_length=150
num_train_epochs=4
warmup_proportion=-1
seed=3306
data_sign=en_onto
checkpoint=28000

gradient_accumulation_steps=4
learning_rate=6e-6
train_batch_size=36
dev_batch_size=72
test_batch_size=72
export_model=/data/export_folder/${train_batch_size}_lr${learning_rate}
output_dir=${export_model}

CUDA_VISIBLE_DEVICES=2 python3 ${base_path}/run/run_relation_extraction.py \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--max_seq_length ${max_seq_length} \
--train_batch_size ${train_batch_size} \
--dev_batch_size ${dev_batch_size} \
--test_batch_size ${test_batch_size} \
--checkpoint ${checkpoint} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--warmup_proportion ${warmup_proportion} \
--export_model ${export_model} \
--output_dir ${output_dir} \
--data_sign ${data_sign} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--allow_impossible 1 
```

## FAQ


