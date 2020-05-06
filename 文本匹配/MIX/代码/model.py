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


logging.info("yw:")
train_yw_word = np.load(mid_path+"train_yw_word.npy")
logging.info(train_yw_word[0])
train_yw_pos = np.load(mid_path+"train_yw_pos.npy")
logging.info(train_yw_pos[0])
train_yw_idf = np.load(mid_path+"train_yw_idf.npy")
logging.info(train_yw_idf[0])
train_yw_position = np.load(mid_path+"train_yw_position.npy")
logging.info(train_yw_position[0])

logging.info("fw1:")
train_fw1_word = np.load(mid_path+"train_fw1_word.npy")
logging.info(train_fw1_word[0])
train_fw1_pos = np.load(mid_path+"train_fw1_pos.npy")
logging.info(train_fw1_pos[0])
train_fw1_idf = np.load(mid_path+"train_fw1_idf.npy")
logging.info(train_fw1_idf[0])
train_fw1_position = np.load(mid_path+"train_fw1_position.npy")
logging.info(train_fw1_position[0])

logging.info("fw2:")
train_fw2_word = np.load(mid_path+"train_fw2_word.npy")
logging.info(train_fw2_word[0])
train_fw2_pos = np.load(mid_path+"train_fw2_pos.npy")
logging.info(train_fw2_pos[0])
train_fw2_idf = np.load(mid_path+"train_fw2_idf.npy")
logging.info(train_fw2_idf[0])
train_fw2_position = np.load(mid_path+"train_fw2_position.npy")
logging.info(train_fw2_position[0])

logging.info("train label:")
train_labels = np.load(mid_path+"train_labels.npy").astype(np.float32)
logging.info(train_labels[0])

dev_yw_word = np.load(mid_path+"dev_yw_word.npy")
dev_yw_pos = np.load(mid_path+"dev_yw_pos.npy")
dev_yw_idf = np.load(mid_path+"dev_yw_idf.npy")
dev_yw_position = np.load(mid_path+"dev_yw_position.npy")
dev_fw1_word = np.load(mid_path+"dev_fw1_word.npy")
dev_fw1_pos = np.load(mid_path+"dev_fw1_pos.npy")
dev_fw1_idf = np.load(mid_path+"dev_fw1_idf.npy")
dev_fw1_position = np.load(mid_path+"dev_fw1_position.npy")
dev_fw2_word = np.load(mid_path+"dev_fw2_word.npy")
dev_fw2_pos = np.load(mid_path+"dev_fw2_pos.npy")
dev_fw2_idf = np.load(mid_path+"dev_fw2_idf.npy")
dev_fw2_position = np.load(mid_path+"dev_fw2_position.npy")
dev_labels = np.load(mid_path+"dev_labels.npy").astype(np.float32)

test_yw_word = np.load(mid_path+"test_yw_word.npy")
test_yw_pos = np.load(mid_path+"test_yw_pos.npy")
test_yw_idf = np.load(mid_path+"test_yw_idf.npy")
test_yw_position = np.load(mid_path+"test_yw_position.npy")
test_fw1_word = np.load(mid_path+"test_fw1_word.npy")
test_fw1_pos = np.load(mid_path+"test_fw1_pos.npy")
test_fw1_idf = np.load(mid_path+"test_fw1_idf.npy")
test_fw1_position = np.load(mid_path+"test_fw1_position.npy")
test_fw2_word = np.load(mid_path+"test_fw2_word.npy")
test_fw2_pos = np.load(mid_path+"test_fw2_pos.npy")
test_fw2_idf = np.load(mid_path+"test_fw2_idf.npy")
test_fw2_position = np.load(mid_path+"test_fw2_position.npy")
test_labels = np.load(mid_path+"test_labels.npy").astype(np.float32)

pos2id=json.load(open(mid_path+"pos2id.json",encoding='utf-8'))
word2id=json.load(open(mid_path+"word2id.json",encoding='utf-8'))

#word2id=json.load(open(mid_path+"word2id.json",encoding='utf-8'))
#embedding_matrix = np.random.randn(len(word2id),300).astype(np.float32)
word_embedding_matrix = np.load(mid_path+"embedding_matrix.npy").astype(np.float32)
pos_embedding_matrix = np.random.randn(len(pos2id),5).astype(np.float32)
posi_embedding_matrix = np.random.randn(150,5).astype(np.float32)

logging.info(train_yw_word.shape)
logging.info(dev_yw_word.shape)
logging.info(test_yw_word.shape)

logging.info(word_embedding_matrix.shape)
logging.info(pos_embedding_matrix.shape)
logging.info(posi_embedding_matrix.shape)

"""
Config
"""
x1_max_len = 150
x2_max_len = 150
x3_max_len = 150

class_nums = 5

epoch_num = 100000
batch_size = 64
lr = 0.001
clip = 5

weight = [30,9,2,2,9]
kernel_num = 50
#weight = [1,1,1,1,1]
isTrain = True

"""
Model
"""
graph = tf.Graph()
with graph.as_default():
    with tf.variable_scope('placeholder'):
        x1 = tf.placeholder(tf.int32, shape=[None, x1_max_len],name="X1")
        x2 = tf.placeholder(tf.int32, shape=[None, x2_max_len],name="X2")

        pos1 = tf.placeholder(tf.int32, shape=[None, x1_max_len], name="POS1")
        pos2 = tf.placeholder(tf.int32, shape=[None, x2_max_len], name="POS2")

        posi1 = tf.placeholder(tf.int32, shape=[None, x1_max_len], name="POSI1")
        posi2 = tf.placeholder(tf.int32, shape=[None, x2_max_len], name="POSI2")

        idf1 = tf.placeholder(tf.float32, shape=[None, x1_max_len], name="IDF1")
        idf2 = tf.placeholder(tf.float32, shape=[None, x2_max_len], name="IDF2")

        labels = tf.placeholder(tf.float32, shape=[None, class_nums], name="labels")

        dropout = tf.placeholder(tf.float32, shape=(),name='dropout')

    with tf.name_scope("word_embedding"):
        word_embedding = tf.get_variable('word_embedding',initializer = word_embedding_matrix, \
                                            dtype=tf.float32, trainable=True)
        word_embed1 = tf.nn.embedding_lookup(word_embedding,x1)
        word_embed2 = tf.nn.embedding_lookup(word_embedding,x2)
    with tf.variable_scope("pos_embedding"):
        pos_embedding = tf.get_variable('pos_embedding',initializer = pos_embedding_matrix, \
                                        dtype=tf.float32, trainable=True)
        pos_embed1 = tf.nn.embedding_lookup(pos_embedding,pos1)
        pos_embed2 = tf.nn.embedding_lookup(pos_embedding,pos2)
    with tf.variable_scope("position_embedding"):
        posi_embedding = tf.get_variable('posi_embedding',initializer = posi_embedding_matrix,
                                        dtype=tf.float32, trainable=True)
        posi_embed1 = tf.nn.embedding_lookup(posi_embedding,posi1)
        posi_embed2 = tf.nn.embedding_lookup(posi_embedding,posi2)
    with tf.name_scope("word_conv1"):
        word1_conv1 = tf.layers.conv1d(inputs=word_embed1,filters=kernel_num,kernel_size=1,strides=1,\
                         padding="same",activation=tf.nn.relu, \
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        word2_conv1 = tf.layers.conv1d(inputs=word_embed2,filters=kernel_num,kernel_size=1,strides=1,\
                         padding="same",activation=tf.nn.relu, \
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.name_scope("word_conv2"):
        word1_conv2 = tf.layers.conv1d(inputs=word_embed1,filters=kernel_num,kernel_size=2,strides=1,\
                         padding="same",activation=tf.nn.relu, \
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        word2_conv2 = tf.layers.conv1d(inputs=word_embed2,filters=kernel_num,kernel_size=2,strides=1,\
                         padding="same",activation=tf.nn.relu, \
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.name_scope("word_conv3"):
        word1_conv3 = tf.layers.conv1d(inputs=word_embed1,filters=kernel_num,kernel_size=3,strides=1,\
                         padding="same",activation=tf.nn.relu, \
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        word2_conv3 = tf.layers.conv1d(inputs=word_embed2,filters=kernel_num,kernel_size=3,strides=1,\
                         padding="same",activation=tf.nn.relu, \
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.name_scope("sim_matrix"):
        sim_matrix_1 = tf.matmul(word1_conv1,tf.transpose(word2_conv1,[0,2,1]))
        sim_matrix_2 = tf.matmul(word1_conv1,tf.transpose(word2_conv2,[0,2,1]))
        sim_matrix_3 = tf.matmul(word1_conv1,tf.transpose(word2_conv3,[0,2,1]))

        sim_matrix_4 = tf.matmul(word1_conv2,tf.transpose(word2_conv1,[0,2,1]))
        sim_matrix_5 = tf.matmul(word1_conv2,tf.transpose(word2_conv2,[0,2,1]))
        sim_matrix_6 = tf.matmul(word1_conv2,tf.transpose(word2_conv3,[0,2,1]))
        
        sim_matrix_7 = tf.matmul(word1_conv3,tf.transpose(word2_conv1,[0,2,1]))
        sim_matrix_8 = tf.matmul(word1_conv3,tf.transpose(word2_conv2,[0,2,1]))
        sim_matrix_9 = tf.matmul(word1_conv3,tf.transpose(word2_conv3,[0,2,1]))
    with tf.name_scope("idf_attention"):
        idf_matrix = tf.matmul(tf.expand_dims(idf1,axis=2), tf.expand_dims(idf2,axis=1))
    with tf.name_scope("pos_attention"):
        pos_matrix = tf.matmul(pos_embed1,tf.transpose(pos_embed2,[0,2,1]))
    with tf.name_scope("position_attention"):
        position_matrix = tf.matmul(posi_embed1,tf.transpose(posi_embed2,[0,2,1]))
    with tf.name_scope("matrix_multiply"):
        matrix1 = tf.multiply(sim_matrix_1,idf_matrix)
        matrix2 = tf.multiply(sim_matrix_1,pos_matrix)
        matrix3 = tf.multiply(sim_matrix_1,position_matrix)
        
        matrix4 = tf.multiply(sim_matrix_2,idf_matrix)
        matrix5 = tf.multiply(sim_matrix_2,pos_matrix)
        matrix6 = tf.multiply(sim_matrix_2,position_matrix)
        
        matrix7 = tf.multiply(sim_matrix_3,idf_matrix)
        matrix8 = tf.multiply(sim_matrix_3,pos_matrix)
        matrix9 = tf.multiply(sim_matrix_3,position_matrix)
        
        matrix10 = tf.multiply(sim_matrix_4,idf_matrix)
        matrix11 = tf.multiply(sim_matrix_4,pos_matrix)
        matrix12 = tf.multiply(sim_matrix_4,position_matrix)
        
        matrix13 = tf.multiply(sim_matrix_5,idf_matrix)
        matrix14 = tf.multiply(sim_matrix_5,pos_matrix)
        matrix15 = tf.multiply(sim_matrix_5,position_matrix)
        
        matrix16 = tf.multiply(sim_matrix_6,idf_matrix)
        matrix17 = tf.multiply(sim_matrix_6,pos_matrix)
        matrix18 = tf.multiply(sim_matrix_6,position_matrix)
        
        matrix19 = tf.multiply(sim_matrix_7,idf_matrix)
        matrix20 = tf.multiply(sim_matrix_7,pos_matrix)
        matrix21 = tf.multiply(sim_matrix_7,position_matrix)
        
        matrix22 = tf.multiply(sim_matrix_8,idf_matrix)
        matrix23 = tf.multiply(sim_matrix_8,pos_matrix)
        matrix24 = tf.multiply(sim_matrix_8,position_matrix)
        
        matrix25 = tf.multiply(sim_matrix_9,idf_matrix)
        matrix26 = tf.multiply(sim_matrix_9,pos_matrix)
        matrix27 = tf.multiply(sim_matrix_9,position_matrix)
    with tf.name_scope("concat_to_cube"):
        cube_data = tf.concat([tf.expand_dims(matrix1,axis=3),tf.expand_dims(matrix2,axis=3),\
                  tf.expand_dims(matrix3,axis=3),tf.expand_dims(matrix4,axis=3),\
                  tf.expand_dims(matrix5,axis=3),tf.expand_dims(matrix6,axis=3),\
                  tf.expand_dims(matrix7,axis=3),tf.expand_dims(matrix8,axis=3),\
                  tf.expand_dims(matrix9,axis=3),tf.expand_dims(matrix10,axis=3),\
                  tf.expand_dims(matrix11,axis=3),tf.expand_dims(matrix12,axis=3),\
                  tf.expand_dims(matrix13,axis=3),tf.expand_dims(matrix14,axis=3),\
                  tf.expand_dims(matrix15,axis=3),tf.expand_dims(matrix16,axis=3),\
                  tf.expand_dims(matrix17,axis=3),tf.expand_dims(matrix18,axis=3),\
                  tf.expand_dims(matrix19,axis=3),tf.expand_dims(matrix20,axis=3),\
                  tf.expand_dims(matrix21,axis=3),tf.expand_dims(matrix22,axis=3),\
                  tf.expand_dims(matrix23,axis=3),tf.expand_dims(matrix24,axis=3),\
                  tf.expand_dims(matrix25,axis=3),tf.expand_dims(matrix26,axis=3),\
                  tf.expand_dims(matrix27,axis=3)],axis=-1)

        logging.info(cube_data)
    with tf.name_scope("conv2d"):
        conv2d = tf.layers.conv2d(cube_data,filters=6,kernel_size=[3,3],padding="same",\
                                    activation=tf.nn.relu,strides=[1,1], \
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        logging.info(conv2d)
        pool2d = tf.layers.max_pooling2d(conv2d,pool_size=[3,3],strides=3,padding="valid")
        logging.info(pool2d)
    with tf.variable_scope('fc1'):
        pool2 = tf.reshape(pool2d, [-1, 50*50*6])
        dropout_layer = tf.nn.dropout(pool2, dropout,name='dropout')
        logging.info(dropout_layer)

        match_1 = tf.split(dropout_layer,2)[0]
        match_2 = tf.split(dropout_layer,2)[1]
        logging.info(match_1)
        logging.info(match_2)

        match = tf.concat([match_1, match_2], -1)
        logging.info(match)

        fc1 = tf.nn.relu(tf.contrib.layers.linear(match, 20))
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
        saver = tf.train.Saver(tf.global_variables())
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
            for idx in get_batch(len(train_yw_word),batch_size):
                yw_word = np.concatenate((train_yw_word[idx], train_yw_word[idx]), axis=0)
                yw_pos = np.concatenate((train_yw_pos[idx], train_yw_pos[idx]), axis=0)
                yw_idf = np.concatenate((train_yw_idf[idx], train_yw_idf[idx]), axis=0)
                yw_posi = np.concatenate((train_yw_position[idx], train_yw_position[idx]), axis=0)

                fw_word = np.concatenate((train_fw1_word[idx], train_fw2_word[idx]), axis=0)
                fw_pos = np.concatenate((train_fw1_pos[idx], train_fw2_pos[idx]), axis=0)
                fw_idf = np.concatenate((train_fw1_idf[idx], train_fw2_idf[idx]), axis=0)
                fw_posi = np.concatenate((train_fw1_position[idx], train_fw2_position[idx]), axis=0)

                label = train_labels[idx]

                _,t_loss,t_acc,t_cm,global_nums,t_pred,t_true = sess.run(
                    [train_op,loss,acc,cm,global_add,pred,true], {
                    x1: yw_word,
                    x2: fw_word,
                    pos1: yw_pos,
                    pos2: fw_pos,
                    posi1: yw_posi,
                    posi2: fw_posi,
                    idf1: yw_idf,
                    idf2: fw_idf,
                    labels : label,
                    dropout: 0.8
                })
                #logging.info("global_nums:%d" % (global_nums))

                if global_nums % 10 == 0:
                    logging.info("Train:global_nums:%d,loss:%f,acc:%f,rho:%f" % \
                                    (global_nums,t_loss,t_acc,pearsonr(t_true, t_pred)[0]))

                if global_nums % 100 == 0:
                    val_pred = []
                    val_label = []
                    for idx in get_batch(len(dev_yw_word),batch_size):
                        yw_word = np.concatenate((dev_yw_word[idx], dev_yw_word[idx]), axis=0)
                        yw_pos = np.concatenate((dev_yw_pos[idx], dev_yw_pos[idx]), axis=0)
                        yw_idf = np.concatenate((dev_yw_idf[idx], dev_yw_idf[idx]), axis=0)
                        yw_posi = np.concatenate((dev_yw_position[idx], dev_yw_position[idx]), axis=0)

                        fw_word = np.concatenate((dev_fw1_word[idx], dev_fw2_word[idx]), axis=0)
                        fw_pos = np.concatenate((dev_fw1_pos[idx], dev_fw2_pos[idx]), axis=0)
                        fw_idf = np.concatenate((dev_fw1_idf[idx], dev_fw2_idf[idx]), axis=0)
                        fw_posi = np.concatenate((dev_fw1_position[idx], dev_fw2_position[idx]), axis=0)

                        label = dev_labels[idx]

                        d_loss,d_pred,d_true = sess.run(
                            [loss,pred,true], {
                            x1: yw_word,
                            x2: fw_word,
                            pos1: yw_pos,
                            pos2: fw_pos,
                            posi1: yw_posi,
                            posi2: fw_posi,
                            idf1: yw_idf,
                            idf2: fw_idf,
                            labels : label,
                            dropout: 1.0
                        })
                        val_pred.extend(d_pred)
                        val_label.extend(d_true)

                    val_rho = pearsonr(val_label, val_pred)[0]
                    logging.info("Dev:global_nums:%d,acc:%f,rho:%f" % \
                                    (global_nums,accuracy_score(val_label, val_pred),val_rho))
                    logging.info("fault tolerant：")
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

        x1 = graph.get_operation_by_name('placeholder/X1').outputs[0]
        x2 = graph.get_operation_by_name('placeholder/X2').outputs[0]

        pos1 = graph.get_operation_by_name('placeholder/POS1').outputs[0]
        pos2 = graph.get_operation_by_name('placeholder/POS2').outputs[0]

        posi1 = graph.get_operation_by_name('placeholder/POSI1').outputs[0]
        posi2 = graph.get_operation_by_name('placeholder/POSI2').outputs[0]

        idf1 = graph.get_operation_by_name('placeholder/IDF1').outputs[0]
        idf2 = graph.get_operation_by_name('placeholder/IDF2').outputs[0]

        labels = graph.get_operation_by_name('placeholder/labels').outputs[0]

        dropout = graph.get_operation_by_name('placeholder/dropout').outputs[0]

        true = graph.get_operation_by_name('evaluation/Cast').outputs[0]
        pred = graph.get_operation_by_name('evaluation/Cast_1').outputs[0]

        test_pred = []
        test_true = []
        for idx in get_batch(len(test_yw_word),batch_size):
            yw_word = np.concatenate((test_yw_word[idx], test_yw_word[idx]), axis=0)
            yw_pos = np.concatenate((test_yw_pos[idx], test_yw_pos[idx]), axis=0)
            yw_idf = np.concatenate((test_yw_idf[idx], test_yw_idf[idx]), axis=0)
            yw_posi = np.concatenate((test_yw_position[idx], test_yw_position[idx]), axis=0)

            fw_word = np.concatenate((test_fw1_word[idx], test_fw2_word[idx]), axis=0)
            fw_pos = np.concatenate((test_fw1_pos[idx], test_fw2_pos[idx]), axis=0)
            fw_idf = np.concatenate((test_fw1_idf[idx], test_fw2_idf[idx]), axis=0)
            fw_posi = np.concatenate((test_fw1_position[idx], test_fw2_position[idx]), axis=0)

            label = dev_labels[idx]
            d_pred,d_true = sess.run(
              [pred,true], {
                x1: yw_word,
                x2: fw_word,
                pos1: yw_pos,
                pos2: fw_pos,
                posi1: yw_posi,
                posi2: fw_posi,
                idf1: yw_idf,
                idf2: fw_idf,
                labels : label,
                dropout: 1.0
            })
            test_pred.extend(d_pred)
            test_true.extend(d_true)

        logging.info(len(test_pred))
        logging.info(len(test_true))
        #bad case analyzes
        logging.info(test_y)
        logging.info(label_y)


        logging.info("Test:acc:%f,rho:%f" % (accuracy_score(test_true,test_pred),pearsonr(test_true,test_pred)[0]))

        logging.info("fault tolerant：")
        logging.info(rongcuodu(test_true,test_pred))
        logging.info("confusion_matrix:")
        cm = confusion_matrix(test_true,test_pred)
        logging.info(cm)





