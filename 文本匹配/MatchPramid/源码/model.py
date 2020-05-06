import tensorflow as tf
import numpy as np
import random
import json
import os
import logging
import sys
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix,accuracy_score


log_dir = 'logdir'
log_file = os.path.join(log_dir, 'log.txt')
if not os.path.isdir(log_dir):
  os.makedirs(log_dir)

logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file))
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

model_path = "./checkpoint"
if not os.path.isdir(model_path):
  os.makedirs(model_path)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 文件路径
path = 'data/'
original_path = path+'original_data/'
assist_path = path+'assist_data/'
mid_path = path+'mid_data/'

train_result_ids = np.load(mid_path+"train_result_ids.npy")
train_fw1_ids = np.load(mid_path+"train_fw1_ids.npy")
train_fw2_ids = np.load(mid_path+"train_fw2_ids.npy")
train_labels = np.load(mid_path+"train_labels.npy").astype(np.float32)

dev_result_ids = np.load(mid_path+"dev_result_ids.npy")
dev_fw1_ids = np.load(mid_path+"dev_fw1_ids.npy")
dev_fw2_ids = np.load(mid_path+"dev_fw2_ids.npy")
dev_labels = np.load(mid_path+"dev_labels.npy").astype(np.float32)

test_result_ids = np.load(mid_path+"test_result_ids.npy")
test_fw1_ids = np.load(mid_path+"test_fw1_ids.npy")
test_fw2_ids = np.load(mid_path+"test_fw2_ids.npy")
test_labels = np.load(mid_path+"test_labels.npy").astype(np.float32)

#word2id=json.load(open(mid_path+"word2id.json",encoding='utf-8'))
#embedding_matrix = np.random.randn(len(word2id),300).astype(np.float32)
embedding_matrix = np.load(mid_path+"embedding_matrix.npy").astype(np.float32)

logging.info(train_result_ids.shape)
logging.info(dev_result_ids.shape)
logging.info(test_result_ids.shape)

logging.info(embedding_matrix.shape)

"""
Config
"""

x1_max_len = 200
x2_max_len = 200
x3_max_len = 200

class_nums = 5

epoch_num = 100000
batch_size = 64
lr = 0.001
clip = 5

weight = [30,9,2,2,9]

#weight = [1,1,1,1,1]
isTrain = True

"""
Model
"""
graph = tf.Graph()
with graph.as_default():
  with tf.variable_scope('placeholder'):
    X1 = tf.placeholder(tf.int32, name='X1',shape=(None, x1_max_len))
    X2 = tf.placeholder(tf.int32, name='X2',shape=(None, x2_max_len))
    X3 = tf.placeholder(tf.int32, name='X3',shape=(None, x3_max_len))
    labels = tf.placeholder(tf.float32, name='Y', shape=(None,class_nums))
    dropout = tf.placeholder(tf.float32, shape=(),name='dropout')
  with tf.variable_scope('embedding'):
    embedding = tf.get_variable('embedding',initializer = embedding_matrix,dtype=tf.float32, trainable=True)

    embed1 = tf.nn.embedding_lookup(embedding, X1)
    embed2 = tf.nn.embedding_lookup(embedding, X2)
    embed3 = tf.nn.embedding_lookup(embedding, X3)

  with tf.variable_scope('mutually'):
    cross1 = tf.einsum('abd,acd->abc', embed1, embed2)
    cross2 = tf.einsum('abd,acd->abc', embed1, embed3)
    logging.info(cross1)
    logging.info(cross2)
    #tf.concat(0, [t1, t2])# ==> [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    cross = tf.concat([cross1, cross2],1)# ==> [[1,2,3,7,8, 9], [4,5,6,10,11, 12]]
    #cross = tf.stack( [cross1,cross1], axis=1)#[?,2,200,200]
    #logging.info(cross)
  with tf.variable_scope('conv1'):   
    cross_img = tf.expand_dims(cross,-1)#tf.reshape(cross, [-1,200,200,2])#[?,200,200,2]
    #cross_img = tf.reshape(cross, [-1,200,200,2])
    #logging.info(cross_img)
    # convolution
    '''
    filters1 = tf.get_variable("filters1", [4, 4, 2, 8], dtype=tf.float32)
    conv1 = tf.nn.depthwise_conv2d(
            input=cross_img,
            filter=filters1,
            strides=[1, 1, 1, 1],
            padding='SAME',
            rate=[1, 1])
    logging.info(conv1)
    conv1_relu = tf.nn.relu(conv1)#[none,20,20,8]
    logging.info(conv1_relu)
    pool1 = tf.nn.max_pool(conv1_relu,[1, 10, 10, 1], [1, 10, 10, 1], "VALID")
    logging.info(pool1)

    filters2 = tf.get_variable("filters2", [3, 3, 16, 1], dtype=tf.float32)
    conv2 = tf.nn.depthwise_conv2d(
            input=pool1,
            filter=filters2,
            strides=[1, 1, 1, 1],
            padding='SAME',
            rate=[1, 1])
    logging.info(conv2)
    conv2_relu = tf.nn.relu(conv2)#[none,20,20,8]
    logging.info(conv2_relu)
    pool2 = tf.nn.max_pool(conv2_relu,[1, 5, 5, 1], [1, 5, 5, 1], "VALID")
    logging.info(pool2)
    '''


    w1 = tf.get_variable('w1',initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.2, dtype=tf.float32),
        dtype=tf.float32, shape=[5, 5, 1, 8])
    b1 = tf.get_variable('b1',initializer=tf.constant_initializer(),
        dtype=tf.float32, shape=[8])

    conv1 = tf.nn.relu(tf.nn.conv2d(cross_img,w1, [1, 1, 1, 1], "SAME") + b1)#[none,20,20,8]
    logging.info(conv1)
    pool1 = tf.nn.max_pool(conv1,[1, 5, 5, 1], [1, 5, 5, 1], "VALID")
    logging.info(pool1)#[?,20,40,8]


  with tf.variable_scope('conv2'): 
    # convolution
    w2 = tf.get_variable('w1',initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.2, dtype=tf.float32),
        dtype=tf.float32, shape=[3, 3, 8, 8])
    b2 = tf.get_variable('b1',initializer=tf.constant_initializer(),
        dtype=tf.float32, shape=[8])

    conv2 = tf.nn.relu(tf.nn.conv2d(pool1,w2, [1, 1, 1, 1], "SAME") + b2)#[none,20,20,8]
    logging.info(conv2)
    pool2 = tf.nn.max_pool(conv2,[1, 5, 5, 1], [1, 5, 5, 1], "VALID")
    logging.info(pool2)

    '''  
    conv2 = tf.nn.conv2d(pool1, [2,2,8,16],[1,1,1,1],padding = "SAME",name='L2_conv')
    conv2 = tf.nn.relu(conv2)
    logging.info(conv2)
    pool2 = tf.nn.max_pool(conv2,[1, 2, 2, 1], [1, 2, 2, 1], "VALID")
    logging.info(pool2)
    '''
  with tf.variable_scope('fc1'):
    pool2 = tf.reshape(pool2, [-1, 8*16*8])
    dropout_layer = tf.nn.dropout(pool2, dropout,name='dropout')
    logging.info(dropout_layer)
    fc1 = tf.nn.relu(tf.contrib.layers.linear(dropout_layer, 20))
    logging.info(fc1)
  with tf.variable_scope("classification"):
    logits = tf.layers.dense(fc1, class_nums,activation=None)
    logging.info(logits)
  #计算损失
  with tf.variable_scope("loss"):
    logits_softmax = tf.nn.softmax(logits)
    #losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
    #loss = tf.reduce_mean(losses)
    loss = (-weight[0] * tf.reduce_mean(labels[:, 0] * tf.log(logits_softmax[:,0]+1e-10))
            -weight[1] * tf.reduce_mean(labels[:, 1] * tf.log(logits_softmax[:,1]+1e-10))
            -weight[2] * tf.reduce_mean(labels[:, 2] * tf.log(logits_softmax[:,2]+1e-10))
            -weight[3] * tf.reduce_mean(labels[:, 3] * tf.log(logits_softmax[:,3]+1e-10))
            -weight[4] * tf.reduce_mean(labels[:, 4] * tf.log(logits_softmax[:,4]+1e-10))
            )
  #选择优化器
  with tf.variable_scope("train_step"):
    global_add = tf.Variable(0, name="global_step", trainable=False)
    #global_add = global_step.assign_add(1)#用于计数

    #train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    update = tf.train.AdamOptimizer(learning_rate=lr)

    grads_and_vars = update.compute_gradients(loss)
    # 对梯度gradients进行裁剪，保证在[-clip, clip]之间。
    grads_and_vars_clip = [[tf.clip_by_value(g, -clip, clip), v] for g, v in grads_and_vars]
    train_op = update.apply_gradients(grads_and_vars_clip, global_step=global_add)

  #准确率/f1/p/r计算
  with tf.variable_scope("evaluation"):
    true = tf.cast(tf.argmax(labels, axis=-1), tf.float32)#真实序列的值
    pred = tf.cast(tf.argmax(logits, axis=-1), tf.float32)#预测序列的值
    print(pred)
    print(true)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32), name="acc")
    cm = tf.contrib.metrics.confusion_matrix(pred, true, num_classes=class_nums)


def rongcuodu(y_true_list, y_pred_list):
    sum_ = []
    num = 0
    num_dict = {0:0, 1:0, 2:0, 3:0, 4:0}
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        sum_.append(abs(y_true - y_pred))
    for i in sum_:
        num_dict[i] += 1
    for k, v in num_dict.items():
        num_dict[k] = v / len(y_true_list)
    return num_dict

def get_batch(total_sample, batch_size = 128, padding=False, shuffle=True):
  data_order = list(range(total_sample))
  if shuffle:
    np.random.shuffle(data_order)
  if padding:
    if total_sample % batch_size != 0:
      data_order += [data_order[-1]] * (batch_size - total_sample % batch_size)
  for i in range(len(data_order) // batch_size):
    idx = data_order[i * batch_size:(i + 1) * batch_size]
    yield idx
  remain = len(data_order) % batch_size
  if remain != 0:
    idx = data_order[-remain:]
    yield idx



with tf.Session(graph=graph) as sess:
  if isTrain:
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=30)
    if not os.path.isdir(model_path):
      os.mkdir(model_path)

    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        logging.info('Loading model from %s', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info('Loading model with fresh parameters')
        init = tf.global_variables_initializer()
        sess.run(init)

    max_rho = -float("inf")
    for epoch in range(epoch_num):
      for idx in get_batch(len(train_result_ids),batch_size):
        _,t_loss,t_acc,t_cm,global_nums,t_pred,t_true = sess.run(
            [train_op,loss,acc,cm,global_add,pred,true], {
            X1: train_result_ids[idx],
            X2: train_fw1_ids[idx],
            X3: train_fw2_ids[idx],
            labels : train_labels[idx],
            dropout: 0.8
        })

        if global_nums % 10 == 0:
          logging.info("Train:global_nums:%d,loss:%f,acc:%f,rho:%f" % \
              (global_nums,t_loss,t_acc,pearsonr(t_true, t_pred)[0]))

        if global_nums % 100 == 0:
          val_pred = []
          val_label = []
          for idx in get_batch(len(dev_result_ids),batch_size):
            d_loss,d_pred,d_true = sess.run(
                [loss,pred,true], {
                X1: dev_result_ids[idx],
                X2: dev_fw1_ids[idx],
                X3: dev_fw2_ids[idx],
                labels : dev_labels[idx],
                dropout: 1.0
            })
            val_pred.extend(d_pred)
            val_label.extend(d_true)

          val_rho = pearsonr(val_label, val_pred)[0]
          logging.info("Dev:global_nums:%d,acc:%f,rho:%f" % \
              (global_nums,accuracy_score(val_label, val_pred),val_rho))
          logging.info("容错：")
          logging.info(rongcuodu(val_label, val_pred))
          logging.info("confusion_matrix:")
          logging.info(confusion_matrix(val_label, val_pred))

          if val_rho > max_rho:
            max_rho = val_rho
            path = saver.save(sess, model_path+'/model.ckpt',global_step=global_nums)
            logging.info("save model:%s" % path)
  else:
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_path)
    selected_ckpt = ckpt.all_model_checkpoint_paths[-1]

    saver = tf.train.import_meta_graph(selected_ckpt+'.meta', graph=graph)
    
    saver.restore(sess,selected_ckpt)

    X1 = graph.get_operation_by_name('placeholder/X1').outputs[0]
    X2 = graph.get_operation_by_name('placeholder/X2').outputs[0]
    X3 = graph.get_operation_by_name('placeholder/X3').outputs[0]
    labels = graph.get_operation_by_name('placeholder/Y').outputs[0]
    dropout = graph.get_operation_by_name('placeholder/dropout').outputs[0]
    true = graph.get_operation_by_name('evaluation/Cast').outputs[0]
    pred = graph.get_operation_by_name('evaluation/Cast_1').outputs[0]

    test_pred = []
    test_true = []
    for idx in get_batch(len(test_result_ids),batch_size):
      d_pred,d_true = sess.run(
          [pred,true], {
          X1: test_result_ids[idx],
          X2: test_fw1_ids[idx],
          X3: test_fw2_ids[idx],
          labels : test_labels[idx],
          dropout: 1.0
      })
      test_pred.extend(d_pred)
      test_true.extend(d_true)

    logging.info(len(test_pred))
    logging.info(len(test_true))
    #bad case analyzes
    #logging.info(test_y)
    #logging.info(label_y)


    logging.info("Test:acc:%f,rho:%f" % (accuracy_score(test_true,test_pred),pearsonr(test_true,test_pred)[0]))

    logging.info("容错：")
    logging.info(rongcuodu(test_true,test_pred))
    logging.info("confusion_matrix:")
    cm = confusion_matrix(test_true,test_pred)
    logging.info(cm)