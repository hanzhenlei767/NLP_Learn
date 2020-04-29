#!/usr/bin/python
# coding=utf8

"""
# Created : 2020/04/26
# Version : python3.7
# Author  : zhenlei.han
# File    : albert_model.py
# Desc    : 
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

from albert import modeling
from albert import optimization
from albert import tokenization
from albert.classifier_utils import *

from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix,accuracy_score

import logging
import sys


log_dir = 'logdir'
log_file = os.path.join(log_dir, 'log.txt')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file))
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

processors = {"PartC": ZWTTProcessor}#修改data_loader.py

class ALBertModel():
    def __init__(self, albert_config, num_labels, seq_length, init_checkpoint):
        self.albert_config = albert_config
        self.num_labels = num_labels
        self.seq_length = seq_length
        self.tower_grads = []
        self.losses = []

        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.batch_size = tf.placeholder(tf.int32,shape=[], name='batch_size')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        print(self.batch_size)
        self.gpu_step = self.batch_size//gpu_nums

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)



        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)

        if num_warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        optimizer = optimization.AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])


        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            pred = []
            label = []
            for d in range(gpu_nums):
                with tf.device("/gpu:%s" % d), tf.name_scope("%s_%s" % ("tower",d)):
                    self.model = modeling.AlbertModel(
                        config=self.albert_config,
                        is_training=self.is_training,
                        input_ids=self.input_ids[d*self.gpu_step:(d+1)*self.gpu_step],
                        input_mask=self.input_mask[d*self.gpu_step:(d+1)*self.gpu_step],
                        token_type_ids=self.segment_ids[d*self.gpu_step:(d+1)*self.gpu_step])
                    print("GPU:", d)

                    tvars = tf.trainable_variables()
                    initialized_variable_names = {}
                    if init_checkpoint:
                        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

                    logging.info("**** Trainable Variables ****")
                    for var in tvars:
                        init_string = ""
                        if var.name in initialized_variable_names:
                            init_string = ", *INIT_FROM_CKPT*"
                        logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

                    output_layer = self.model.get_pooled_output()
                    logging.info(output_layer)

                    if self.is_training == True:
                        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)


                    match_1 = tf.strided_slice(output_layer, [0], [self.gpu_step], [2])
                    match_2 = tf.strided_slice(output_layer, [1], [self.gpu_step], [2])

                    match = tf.concat([match_1, match_2], 1)

                    self.logits = tf.layers.dense(match, self.num_labels, name='fc', reuse=tf.AUTO_REUSE)


                    logging.info(self.logits)
                    self.r_labels = tf.strided_slice(self.labels[d*self.gpu_step:(d+1)*self.gpu_step], [0], [self.gpu_step], [2])
                    logging.info(self.r_labels)

                    self.r_labels = tf.expand_dims(self.r_labels, -1)
                    logging.info(self.r_labels)
                    self.loss = tf.losses.mean_squared_error(self.logits, self.r_labels)

                    tvars = tf.trainable_variables()
                    grads = tf.gradients(self.loss, tvars)

                    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

                    self.tower_grads.append(list(zip(grads, tvars)))
                    self.losses.append(self.loss)
                    label.append(self.r_labels)
                    pred.append(self.logits)
                outer_scope.reuse_variables()


        with tf.name_scope("apply_gradients"), tf.device("/cpu:0"):
            gradients = self.average_gradients(self.tower_grads)
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)
            new_global_step = global_step + 1
            self.train_op = tf.group(train_op, [global_step.assign(new_global_step)])
            self.losses = tf.reduce_mean(self.losses)
            self.pred = tf.concat(pred, 0)
            self.label = tf.concat(label, 0)
            logging.info(self.pred)
            logging.info(self.label)






            #self.optim = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps, False)


    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
            over the devices. The inner list ranges over the different variables.
        Returns:
                List of pairs of (gradient, variable) where the gradient has been averaged
                across all towers.
        """
        # calculate average gradient for each shared variable across all GPUs
        average_grads = []
        
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            # We need to average the gradients across each GPU.
            
            g0, v0 = grad_and_vars[0]
            if isinstance(g0, tf.IndexedSlices):
                # If the gradient is type IndexedSlices then this is a sparse
                # gradient with attributes indices and values.
                # To average, need to concat them individually then create
                # a new IndexedSlices object.
                indices = []
                values = []
                for g, v in grad_and_vars:
                    indices.append(g.indices)
                    values.append(g.values)
                all_indices = tf.concat(indices, 0)
                avg_values = tf.concat(values, 0) / len(grad_and_vars)
                # deduplicate across indices
                av, ai = self._deduplicate_indexed_slices(avg_values, all_indices)
                grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

            else:
                # a normal tensor can just do a simple average
                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                grads = [g for g, _ in grad_and_vars]
                grad = tf.reduce_mean(grads, 0)

            # the Variables are redundant because they are shared
            # across towers. So.. just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _deduplicate_indexed_slices(self, values, indices):
        """Sums `values` associated with any non-unique `indices`.
        Args:
          values: A `Tensor` with rank >= 1.
          indices: A one-dimensional integer `Tensor`, indexing into the first
          dimension of `values` (as in an IndexedSlices object).
        Returns:
          A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
          de-duplicated version of `indices` and `summed_values` contains the sum of
          `values` slices associated with each unique index.
        """
        unique_indices, new_index_positions = tf.unique(indices)
        summed_values = tf.unsorted_segment_sum(values, new_index_positions, tf.shape(unique_indices)[0])
        return (summed_values, unique_indices)


def make_tf_record(output_dir, data_dir, vocab_file, spm_model_file):
    tf.gfile.MakeDirs(output_dir)       #"model/bert"
    processor = processors[task_name]() #"atec"
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,spm_model_file=spm_model_file)
    train_file = os.path.join(output_dir, "train.tf_record")
    eval_file = os.path.join(output_dir, "eval.tf_record")

    # save data to tf_record
    if not os.path.isfile(train_file):
        train_examples = processor.get_train_examples(data_dir)
        file_based_convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_file,task_name)
        del train_examples
    # eval data
    if not os.path.isfile(eval_file):
        eval_examples = processor.get_dev_examples(data_dir)
        file_based_convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, eval_file,task_name)
        del eval_examples


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def read_data(data, batch_size, is_training, num_epochs):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.

    if is_training:
        data = data.shuffle(buffer_size=500000)
        data = data.repeat(num_epochs)


    data = data.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
    return data

def get_test_features(examples, label_list, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,max_seq_length, tokenizer,task_name)

        features.append(feature)

    return features


def get_test_example():
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,spm_model_file=spm_model_file)
    # save data to tf_record
    #test_examples = processor.get_test_examples("")#测试数据目录
    test_examples = processor.get_test_examples(data_dir)
    features = get_test_features(test_examples, label_list, max_seq_length, tokenizer)

    return features


def evaluate(sess, model):
    """
    评估 val data 的准确率和损失
    """
    # dev data
    test_record = tf.data.TFRecordDataset("./model/eval.tf_record")
    test_data = read_data(test_record, train_batch_size, False, 1)
    test_iterator = test_data.make_one_shot_iterator()
    test_batch = test_iterator.get_next()

    ds_pred = []
    ds_label = []
    while True:
        try:
            features = sess.run(test_batch)
            feed_dict = {model.input_ids: features["input_ids"],
                         model.input_mask: features["input_mask"],
                         model.segment_ids: features["segment_ids"],
                         model.labels: features["label_ids"],
                         model.batch_size: train_batch_size,
                         model.is_training: False}
            d_pred,d_labels = sess.run([model.pred,model.label], feed_dict=feed_dict)
            ds_pred.extend(d_pred)
            ds_label.extend(d_labels)
        except Exception as e:
            #print("Dev error:")
            #print(e)
            break
    return ds_pred,ds_label


def rongcuodu(y_test, y_pre, score = 24):
    et_005_num = 0
    et_010_num = 0
    et_015_num = 0
    et_020_num = 0
    
    et005_score = score * 0.05
    et010_score = score * 0.10
    et015_score = score * 0.15
    et020_score = score * 0.20
    #回归问题范围
    y_test= np.array(y_test)
    y_test[y_test > score] = score
    y_test[y_test < 0] = 0
    diff = np.abs(np.array(y_pre)-y_test)
    for i in diff:
        
        if i <= et005_score:
            et_005_num = et_005_num + 1
        elif i <= et010_score:
            et_010_num = et_010_num + 1
        elif i <= et015_score:
            et_015_num = et_015_num + 1
        elif i <= et020_score:
            et_020_num = et_020_num + 1
        else:
            continue
    rows = len(y_test)
    ratio_005 = et_005_num / rows
    ratio_010 = et_010_num / rows
    ratio_015 = et_015_num / rows
    ratio_020 = et_020_num / rows
    ratio_num = et_005_num + et_010_num + et_015_num + et_020_num
    ratio = ratio_num / rows

    return ratio


def main():
    albert_config = modeling.AlbertConfig.from_json_file(albert_config_file)
    with tf.Graph().as_default():
        # train data
        train_record = tf.data.TFRecordDataset("./model/train.tf_record")
        train_data = read_data(train_record, train_batch_size, True, num_train_epochs)
        train_iterator = train_data.make_one_shot_iterator()

        model = ALBertModel(albert_config, num_labels, max_seq_length, init_checkpoint)
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=160)
        train_steps = 0
        val_loss = 0.0
        val_acc = 0.0
        max_et = 0.0
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            train_batch = train_iterator.get_next()
            while True:
                try:
                    train_steps += 1
                    features = sess.run(train_batch)
                    feed_dict = {model.input_ids: features["input_ids"],
                                 model.input_mask: features["input_mask"],
                                 model.segment_ids: features["segment_ids"],
                                 model.labels: features["label_ids"],
                                 model.batch_size: train_batch_size,
                                 model.is_training: True}
                    _, train_loss,train_pred,train_labels = \
                    sess.run([model.train_op, model.loss,model.pred,model.label],
                        feed_dict=feed_dict)
                    if train_steps % 5 == 0:
                        #logging.info(train_labels)
                        #logging.info(train_pred)
                        logging.info("Train:global_nums:%d,loss:%f,et:%f,rho:%f" % \
                                (train_steps,train_loss, \
                                rongcuodu(train_labels, train_pred),pearsonr(train_labels, train_pred)[0]))

                    if train_steps % 20 == 0:
                        val_pred,val_label = evaluate(sess, model)
                        val_rho = pearsonr(val_label, val_pred)[0]
                        val_et = rongcuodu(val_label, val_pred)
                        logging.info("Dev:global_nums:%d,et:%f,rho:%f" % (train_steps,val_et,val_rho))
                        if max_et <= val_et:
                            max_et = val_et
                            path = saver.save(sess, "./model/bert/model", global_step=train_steps)
                            logging.info("save model:%s" % path)
                except Exception as e:
                    #logging.info("train error:")
                    #logging.info(e)
                    break


def test_model(sess, graph, features):
    """

    :param sess:
    :param graph:
    :param features:
    :return:
    """
    input_ids = graph.get_operation_by_name('input_ids').outputs[0]
    input_mask = graph.get_operation_by_name('input_mask').outputs[0]
    segment_ids = graph.get_operation_by_name('segment_ids').outputs[0]
    labels = graph.get_operation_by_name('labels').outputs[0]
    batch_size = graph.get_operation_by_name('batch_size_1').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]

    pred = graph.get_operation_by_name('apply_gradients/concat_2').outputs[0]
    r_labels = graph.get_operation_by_name('apply_gradients/concat_3').outputs[0]

    data_len = len(features)
    bs = 8
    num_batch = int((len(features) - 1) / bs) + 1 #巧妙

    test_y = []
    label_y = []
    print(len(features),bs)

    for i in range(num_batch):
        start_index = i * bs
        end_index = min((i + 1) * bs, data_len)

        _input_ids = np.array([data.input_ids for data in features[start_index:end_index]])
        _input_mask = np.array([data.input_mask for data in features[start_index:end_index]])
        _segment_ids = np.array([data.segment_ids for data in features[start_index:end_index]])
        _labels = np.array([data.label_id for data in features[start_index:end_index]])
        feed_dict = {input_ids: _input_ids,
                     input_mask: _input_mask,
                     segment_ids: _segment_ids,
                     labels: _labels,
                     batch_size: bs,
                     is_training: False}
        #print("1")
        pred_y,r_label_y = sess.run([pred,r_labels], feed_dict=feed_dict)
        #print("2")
        test_y.extend(pred_y)
        label_y.extend(r_label_y)
    #logging.info(test_y)
    return test_y,label_y



def test():
    features = get_test_example()
    model_path = "./model/bert"
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_path)
    selected_ckpt = ckpt.all_model_checkpoint_paths[-1]

    saver = tf.train.import_meta_graph(selected_ckpt+'.meta', graph=graph)
    
    #saver.restore(sess, tf.train.latest_checkpoint(model_path))
    saver.restore(sess,selected_ckpt)


    test_y,label_y = test_model(sess, graph, features)
    logging.info(len(label_y))
    logging.info(len(test_y))
    #bad case analyzes
    logging.info(np.array(test_y).reshape(len(test_y),))
    logging.info(np.array(label_y).reshape(len(label_y),))
    #'''
    x_data = pd.DataFrame()
    x_data["label"] = np.array(label_y).reshape(len(label_y),)
    x_data["pred"] = np.array(test_y).reshape(len(test_y),)
    x_data.to_csv("zh800.csv")
    #'''
    logging.info("Test:rho:%f" % (pearsonr(label_y, test_y)[0]))

    logging.info("容错：")
    logging.info(rongcuodu(label_y, test_y))

    '''

    test_df = pd.read_csv(sys.argv[1], sep='\t', header=None, names=["index", "s1", "s2"])
    
    test_index = np.array(test_df["index"])

    with open(sys.argv[2], 'w') as fout:
        for index,pre in enumerate(test_y):
            if pre >=0.5:
                fout.write(str(test_index[index]) + '\t1\n')
            else:
                fout.write(str(test_index[index]) + '\t0\n')
    '''
if __name__ == "__main__":
    data_dir = "data/PartC"
    output_dir = "model"
    task_name = "PartC"
    vocab_file = "./albert/albert_base/30k-clean.vocab"
    spm_model_file = "./albert/albert_base/30k-clean.model"
    albert_config_file = "./albert/albert_base/albert_config.json"
    init_checkpoint = "./albert/albert_base/model.ckpt-best"
    max_seq_length = 380
    init_lr = 2e-5
    #learning_rate = 2e-5
    train_batch_size = 32
    num_train_epochs = 4
    warmup_proportion = 0.1
    num_train_steps = int(21970 / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    num_labels = 1
    gpu_nums = 4
    make_tf_record(output_dir, data_dir, vocab_file,spm_model_file)
    main()
    #test()

