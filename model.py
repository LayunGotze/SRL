import math
import helper
import numpy as np
import tensorflow as tf

class BiLSTM_CRF(object):

    def __init__(self,num_chars, num_poses, num_dises, num_classes, num_steps=200, num_epochs=10):
        # 训练参数
        self.max_f1 = 0
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 64
        self.num_layers = 1   
        self.emb_dim = 50 #char, left, right, rel
        self.pos_dim = 25 #pos, lpos, rpos
        self.dis_dim = 25 #dis
        self.hidden_dim = 300
        self.filter_sizes = [3]
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_poses = num_poses
        self.num_dises = num_dises
        self.num_classes = num_classes

        #特征的placeholder
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.lefts = tf.placeholder(tf.int32, [None, self.num_steps])
        self.rights = tf.placeholder(tf.int32, [None, self.num_steps])
        self.poses = tf.placeholder(tf.int32, [None, self.num_steps])
        self.lposes = tf.placeholder(tf.int32, [None, self.num_steps])
        self.rposes = tf.placeholder(tf.int32, [None, self.num_steps])
        self.rels = tf.placeholder(tf.int32, [None, self.num_steps])
        self.dises = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])

        #word embedding
        self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])

        #多层lstm
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

    def train(self,train_data):
        # TODO 验证集
        X_train = train_data['char']
        X_left_train = train_data['left']
        X_right_train = train_data['right']
        X_pos_train = train_data['pos']
        X_lpos_train = train_data['lpos']
        X_rpos_train = train_data['rpos']
        X_rel_train = train_data['rel']
        X_dis_train = train_data['dis']
        y_train = train_data['label']

        char2id, id2char = helper.loadMap("char2id")
        pos2id, id2pos = helper.loadMap("pos2id")
        label2id, id2label = helper.loadMap("label2id")

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size)) #每轮次数




        