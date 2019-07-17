import re
import csv
import numpy as np
import pandas as pd

#列名： 单词 左侧单词 右侧单词 当前词词性 左词词性 右词词性 谓语词 与谓语词的距离 srl标签
csv_name = ["char", "left", "right", "pos", "lpos", "rpos", "rel", "dis", "label"]
#train_path="data/train.in"
train_path="data/validation.in" #暂时读取小文件

def build_map(train_path):
    #读取训练文件，为单词、词性和SRL标注构建词典，并写入文件中
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=csv_name)

    chars = list(set(df_train["char"][df_train["char"].notnull()]))
    poses = list(set(df_train["pos"][df_train["pos"].notnull()]))
    labels = list(set(df_train["label"][df_train["label"].notnull()]))

    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    pos2id = dict(zip(poses, range(1, len(poses) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))

    char2id["<PAD>"] = 0
    pos2id["<PAD>"] = 0
    label2id["<PAD>"] = 0

    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2pos = dict(zip(range(1, len(poses) + 1), poses))
    id2label = dict(zip(range(1, len(labels) + 1), labels))

    id2char[0] = "<PAD>"
    id2pos[0] = "<PAD>"
    id2label[0] = "<PAD>"

    #将组成字典写入文件
    with open("data/char2id.in", "wb") as outfile:
        for idx in id2char:
            outfile.write((id2char[idx] + "\t" + str(idx) + "\r\n").encode())
    with open("data/pos2id.in", "wb") as outfile:
        for idx in id2pos:
            outfile.write((id2pos[idx] + "\t" + str(idx) + "\r\n").encode())
    with open("data/label2id.in", "wb") as outfile:
        for idx in id2label:
            outfile.write((id2label[idx] + "\t" + str(idx) + "\r\n").encode())
    
    return char2id,pos2id,label2id,id2char,id2pos,id2label
#build_map(train_path)


def load_map(token2id_filepath):
    #将字典文件加载
    token2id = {}
    id2token = {}
    with open(token2id_filepath,'r',encoding='utf-8') as infile:
        for row in infile:
            row = row.rstrip()
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token
#pos2id,id2pos=loadMap('data/pos2id.in')
#print(pos2id)

def padding(sample, seq_max_len):
    #对每个句子做对齐，长度为seq_max_len
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            #短句补0
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
        else:
            #长句截取
            sample[i]=sample[i][:seq_max_len]
    return sample


def prepare(chars, lefts, rights, poss, lposs, rposs, rels, diss, labels, seq_max_len, is_padding=True):
    #将输入数据划分为句子，并且对每句话做padding
    X = []
    X_left = []
    X_right = []
    X_pos = []
    X_lpos = []
    X_rpos = []
    X_rel = []
    X_dis = []
    y = [] #最终变量，存储所有句子向量

    tmp_x = []
    tmp_left = []
    tmp_right = []
    tmp_pos = []
    tmp_lpos = []
    tmp_rpos =[]
    tmp_rel = []
    tmp_dis = []
    tmp_y = [] #存储每个句子的向量

    for record in zip(chars, lefts, rights, poss, lposs, rposs, rels, diss, labels):
        c = record[0]
        lc = record[1]
        rc = record[2]
        p = record[3]
        lp = record[4]
        rp = record[5]
        rl = record[6]
        d = record[7]
        l = record[8]

        if c == -1:
            #如果是句尾，将句子向量放入总向量中
            X.append(tmp_x)
            X_left.append(tmp_left)
            X_right.append(tmp_right)
            X_pos.append(tmp_pos)
            X_lpos.append(tmp_lpos)
            X_rpos.append(tmp_rpos)
            X_rel.append(tmp_rel)
            X_dis.append(tmp_dis)
            y.append(tmp_y)
            tmp_x = []
            tmp_left = []
            tmp_right = []
            tmp_pos = []
            tmp_lpos = []
            tmp_rpos =[]
            tmp_rel = []
            tmp_dis = []
            tmp_y = []
        else:
            #如果非句尾，将此单词信息加入句子向量
            tmp_x.append(c)
            tmp_left.append(lc)
            tmp_right.append(rc)
            tmp_pos.append(p)
            tmp_lpos.append(lp)
            tmp_rpos.append(rp)
            tmp_rel.append(rl)
            tmp_dis.append(int(d))
            tmp_y.append(l)
    #转换为numpy数组
    if is_padding:
        X = np.array(padding(X, seq_max_len))
        X_left = np.array(padding(X_left, seq_max_len))
        X_right = np.array(padding(X_right, seq_max_len))
        X_pos = np.array(padding(X_pos, seq_max_len))
        X_lpos = np.array(padding(X_lpos, seq_max_len))
        X_rpos = np.array(padding(X_rpos, seq_max_len))
        X_rel = np.array(padding(X_rel, seq_max_len))
        X_dis = np.array(padding(X_dis, seq_max_len))
    else:
        X = np.array(X)
        X_left = np.array(X_left)
        X_right = np.array(X_right)
        X_pos = np.array(X_pos)
        X_lpos = np.array(X_lpos)
        X_rpos = np.array(X_rpos)
        X_rel = np.array(X_rel)
        X_dis = np.array(X_dis)
    y = np.array(padding(y, seq_max_len))
    return X, X_left, X_right, X_pos, X_lpos, X_rpos, X_rel, X_dis, y


def get_train(train_path):
    char2id, id2char = load_map('data/char2id.in')
    pos2id, id2pos = load_map('data/pos2id.in')
    label2id, id2label = load_map('data/label2id.in')

    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=csv_name)

    #将输入文件转换为id
    df_train["char_id"] = df_train.char.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["left_id"] = df_train.left.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["right_id"] = df_train.right.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["rel_id"] = df_train.rel.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])

    df_train["pos_id"] = df_train.pos.map(lambda x: -1 if str(x) == str(np.nan) else pos2id[x])
    df_train["lpos_id"] = df_train.lpos.map(lambda x: -1 if str(x) == str(np.nan) else pos2id[x])
    df_train["rpos_id"] = df_train.rpos.map(lambda x: -1 if str(x) == str(np.nan) else pos2id[x])

    df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else label2id[x])


    seq_max_len=200
    X, X_left, X_right, X_pos, X_lpos, X_rpos, X_rel, X_dis, y = prepare(df_train["char_id"], df_train["left_id"], df_train["right_id"],
            df_train["pos_id"], df_train["lpos_id"], df_train["rpos_id"],
            df_train["rel_id"], df_train["dis"], df_train["label_id"], seq_max_len,1000)
    print(X.shape)

    #打乱输入数据
    num_samples = len(X)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    X = X[indexs]
    X_left = X_left[indexs]
    X_right = X_right[indexs]
    X_pos = X_pos[indexs]
    X_lpos = X_lpos[indexs]
    X_rpos = X_rpos[indexs]
    X_rel = X_rel[indexs]
    X_dis = X_dis[indexs]
    y = y[indexs]


    #未处理验证集
    X_train = X
    X_left_train = X_left
    X_right_train = X_right
    X_pos_train = X_pos
    X_lpos_train = X_lpos
    X_rpos_train = X_rpos
    X_rel_train = X_rel
    X_dis_train = X_dis
    y_train = y

    train_data = {}
    train_data['char'] = X_train
    train_data['left'] = X_left_train
    train_data['right'] = X_right_train
    train_data['pos'] = X_pos_train
    train_data['lpos'] = X_lpos_train
    train_data['rpos'] = X_rpos_train
    train_data['rel'] = X_rel_train
    train_data['dis'] = X_dis_train
    train_data['label'] = y_train
    
    return train_data

get_train(train_path)